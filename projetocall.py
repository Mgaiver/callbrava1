import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
from datetime import date, timedelta, datetime
from io import BytesIO
import pytz

# =================================================================
# CONSTANTES E CONFIGURAÇÕES
# =================================================================
RSI_PERIOD = 9
MA_SHORT = 21
MA_LONG = 50
PROXIMITY_TOLERANCE = 0.015
TIMEZONE = pytz.timezone('America/Sao_Paulo')

# Lista de ativos populares para facilitar a seleção do usuário
ASSET_LIST = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "WEGE3",
    "MGLU3", "VIIA3", "B3SA3", "SUZB3", "GGBR4", "JBSS3"
]

# =================================================================
# FUNÇÕES DE PROCESSAMENTO E ANÁLISE DE DADOS
# =================================================================

@st.cache_data(ttl=900) # Cache de 15 minutos
def carregar_e_processar_dados(ticker: str, start_date: date, end_date: date):
    """
    Baixa os dados do ativo, calcula os indicadores técnicos e retorna um DataFrame.
    Retorna um DataFrame processado ou None em caso de erro.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            st.error(f"Não foram encontrados dados para o ativo **{ticker.replace('.SA', '')}** no período selecionado.")
            return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar os dados: {e}")
        return None

    # --- LÓGICA DE LIMPEZA DE COLUNAS ROBUSTA ---
    # yfinance pode retornar colunas como MultiIndex (tuplas) ou strings.
    # Esta lógica lida com ambos os casos para extrair os nomes corretos.
    cleaned_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Se for uma tupla como ('High', 'PETR4.SA'), pegamos o primeiro elemento 'High'
            cleaned_columns.append(col[0])
        else:
            # Se for uma string, apenas a usamos
            cleaned_columns.append(str(col))
    df.columns = cleaned_columns

    # Garante que as colunas estão com nomes padronizados (ex: 'open' -> 'Open')
    try:
        df.columns = [col.strip().title() for col in df.columns]
    except Exception as e:
        st.error(f"Não foi possível padronizar os nomes das colunas. Erro: {e}")
        st.warning(f"Colunas recebidas do provedor de dados: {list(df.columns)}")
        return None

    # Verifica se as colunas essenciais para o cálculo existem
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error("Os dados recebidos não contêm as colunas essenciais (High, Low, Close, Volume) após a padronização.")
        st.info(f"Colunas encontradas: {list(df.columns)}")
        return None

    # Calcula Pontos de Pivô Clássico
    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])

    # Calcula IFR (RSI)
    df.ta.rsi(length=RSI_PERIOD, append=True)
    df.rename(columns={f'RSI_{RSI_PERIOD}': 'RSI'}, inplace=True)

    # Calcula Média Móvel de Volume
    df['Volume_MA'] = df['Volume'].rolling(window=MA_SHORT).mean()

    return df

def gerar_relatorio_analise(df: pd.DataFrame, ticker_name: str):
    """
    Gera um relatório estruturado com análise técnica, recomendação e níveis de preço.
    """
    if df is None or df.empty:
        return "Erro: DataFrame vazio ou inválido."

    dados_atuais = df.iloc[-1]
    fechamento = dados_atuais['Close']
    data_fechamento = df.index[-1].strftime('%d/%m/%Y')
    hora_analise = datetime.now(TIMEZONE).strftime('%d/%m/%Y às %H:%M:%S')

    # Checagem de segurança para indicadores
    ifr = dados_atuais.get('RSI')
    r1 = dados_atuais.get('R1')
    s1 = dados_atuais.get('S1')
    r2 = dados_atuais.get('R2')
    s2 = dados_atuais.get('S2')
    pp = dados_atuais.get('PP')

    if any(v is None for v in [ifr, r1, s1, r2, s2, pp]):
        return "Erro: Não foi possível calcular todos os indicadores necessários."

    # --- 1. Determinação da Recomendação ---
    recomendacao_acao = "**NEUTRA / AGUARDAR**"
    nivel_entrada = 0.0
    justificativa_ifr = f"IFR({RSI_PERIOD}) atual ({ifr:.2f}) indica condições neutras."

    if fechamento <= s1 + (s1 * PROXIMITY_TOLERANCE) and ifr < 35:
        recomendacao_acao = "**COMPRA AGRESSIVA / LONG**"
        nivel_entrada = s1
        justificativa_ifr = f"IFR({RSI_PERIOD}) ({ifr:.2f}) está em zona de **Sobre-Venda (< 35)**."
    elif fechamento >= r1 - (r1 * PROXIMITY_TOLERANCE) and ifr > 65:
        recomendacao_acao = "**VENDA AGRESSIVA / SHORT**"
        nivel_entrada = r1
        justificativa_ifr = f"IFR({RSI_PERIOD}) ({ifr:.2f}) está em zona de **Sobre-Compra (> 65)**."
    elif fechamento <= s1:
        recomendacao_acao = "**COMPRA MODERADA**"
        nivel_entrada = s1
    elif fechamento >= r1:
        recomendacao_acao = "**VENDA MODERADA**"
        nivel_entrada = r1

    # --- 2. Montagem do Relatório Estruturado ---
    relatorio = f"### Análise para {ticker_name.replace('.SA', '')}\n"
    relatorio += f"**Preço de Fechamento:** R$ {fechamento:.2f} (em {data_fechamento})\n"
    relatorio += f"**Análise Gerada em:** {hora_analise} (Fonte: Yahoo Finance)\n\n"

    relatorio += f"**Ponto de Pivô (PP):** R$ {pp:.2f}\n"
    relatorio += f"**Suporte Imediato (S1):** R$ {s1:.2f}\n"
    relatorio += f"**Resistência Imediata (R1):** R$ {r1:.2f}\n\n"

    relatorio += "#### Análise da Tendência e IFR\n"
    relatorio += f"O ativo está entre o Suporte 1 (S1) e a Resistência 1 (R1).\n"
    relatorio += f"- **Índice de Força Relativa:** {justificativa_ifr}\n\n"

    relatorio += "#### Recomendação de Ação\n"
    if recomendacao_acao != "**NEUTRA / AGUARDAR**":
        acao_verbo = "COMPRAR" if "COMPRA" in recomendacao_acao else "VENDER"
        relatorio += f"Com base na combinação de preço e IFR, a recomendação é de **{recomendacao_acao}**.\n"
        relatorio += f"Ponto de entrada sugerido: **R$ {nivel_entrada:.2f}** (operar próximo da zona de reversão).\n"
    else:
        relatorio += "**Aguardar:** Não há um sinal claro. Recomenda-se esperar o preço se aproximar de S1 ou R1.\n\n"

    relatorio += "---\n"
    relatorio += f"**Próximos Níveis:** Suporte 2 (R$ {s2:.2f}) e Resistência 2 (R$ {r2:.2f})."

    return relatorio

# =================================================================
# FUNÇÕES DE PLOTAGEM
# =================================================================

def plotar_grafico(df: pd.DataFrame, ativo_nome: str, tema: str):
    """
    Gera o gráfico e retorna-o como um objeto de bytes para o Streamlit.
    """
    if df is None or df.empty:
        return None

    # Configuração de estilo com base no tema
    if tema == 'Escuro':
        style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle=':', y_on_right=False,
                                   rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
    else:
        style = mpf.make_mpf_style(base_mpf_style='yahoo', gridstyle=':', y_on_right=False)

    # Níveis de Pivô
    ultimo_dia = df.iloc[-1]
    pivot_levels = [
        (ultimo_dia['PP'], 'gray', '-', 0.8),
        (ultimo_dia['S1'], 'green', '--', 1.2),
        (ultimo_dia['R1'], 'red', '--', 1.2),
        (ultimo_dia['S2'], 'darkgreen', ':', 0.8),
        (ultimo_dia['R2'], 'darkred', ':', 0.8),
    ]
    hlines_pivots = [p[0] for p in pivot_levels]
    colors_pivots = [p[1] for p in pivot_levels]
    styles_pivots = [p[2] for p in pivot_levels]
    widths_pivots = [p[3] for p in pivot_levels]

    # Painel do IFR
    add_plots = []
    if 'RSI' in df.columns:
        add_plots.extend([
            mpf.make_addplot(df['RSI'], panel=2, color='blue', ylabel=f'IFR({RSI_PERIOD})', ylim=(0, 100)),
            mpf.make_addplot([70] * len(df), panel=2, color='red', linestyle='-.', width=0.7),
            mpf.make_addplot([30] * len(df), panel=2, color='green', linestyle='-.', width=0.7)
        ])

    buf = BytesIO()
    fig, _ = mpf.plot(
        df,
        type='candle',
        style=style,
        title=f"\nAnálise Técnica: {ativo_nome}",
        ylabel='Preço (R$)',
        volume=True,
        mav=(MA_SHORT, MA_LONG),
        addplot=add_plots if add_plots else None,
        hlines=dict(hlines=hlines_pivots, colors=colors_pivots, linestyle=styles_pivots, linewidths=widths_pivots),
        show_nontrading=False,
        figscale=1.8,
        figratio=(12, 7),
        panel_ratios=(5, 2),
        volume_panel=1,
        ylabel_lower='Volume',
        returnfig=True,
        tight_layout=True,
        update_width_config=dict(candle_linewidth=1.0)
    )

    # Adiciona a marca d'água
    fig.text(0.5, 0.5, 'Brava', fontsize=60, color='gray', ha='center', va='center', alpha=0.15)

    fig.savefig(buf, format='png', bbox_inches='tight')
    return buf.getvalue()

# =================================================================
# INTERFACE PRINCIPAL DO STREAMLIT
# =================================================================

def main():
    """
    Função principal que executa a aplicação Streamlit.
    """
    st.set_page_config(page_title="Call Brava", layout="wide")

    # --- BARRA LATERAL (INPUTS) ---
    with st.sidebar:
        st.image("https://i.imgur.com/vEpA2nO.png", width=70)
        st.title("Configurações da Análise")

        # Seleção de ativo
        ativo_popular = st.selectbox(
            "Selecione um ativo popular:",
            options=sorted(ASSET_LIST),
            index=0
        )
        ativo_custom = st.text_input("Ou digite um código (Ex: VIIA3):").upper().strip()

        ativo_input = ativo_custom if ativo_custom else ativo_popular

        # Seleção de período
        periodo_map = {"Últimos 3 meses": 90, "Últimos 6 meses": 180, "Último ano": 365}
        periodo_selecionado = st.selectbox(
            "Período de Análise:",
            options=list(periodo_map.keys()),
            index=1
        )
        dias = periodo_map[periodo_selecionado]

        # Seleção de tema
        thema_grafico = st.radio("Tema do Gráfico:", ('Claro', 'Escuro'))

        # Botão de execução
        run_button = st.button("Executar Análise", type="primary", use_container_width=True)

    # --- ÁREA PRINCIPAL (OUTPUTS) ---
    st.title("📈 Call Brava")
    st.markdown("Análise técnica simplificada para ativos da B3, baseada em Pontos de Pivô e IFR.")

    if run_button:
        if not ativo_input:
            st.warning("Por favor, selecione ou digite um código de ativo.")
            return

        ticker_yf = ativo_input if ativo_input.endswith('.SA') else f"{ativo_input}.SA"

        with st.spinner(f"Analisando {ativo_input}..."):
            df_processado = carregar_e_processar_dados(ticker_yf, date.today() - timedelta(days=dias), date.today())

            if df_processado is not None and not df_processado.empty:
                relatorio_texto = gerar_relatorio_analise(df_processado, ticker_yf)
                fig_bytes = plotar_grafico(df_processado, ativo_input, thema_grafico)

                st.subheader(f"Resultado da Análise para {ativo_input}")
                tab1, tab2 = st.tabs(["📊 Relatório de Análise", "📈 Gráfico Técnico"])

                with tab1:
                    st.markdown(relatorio_texto, unsafe_allow_html=True)

                with tab2:
                    if fig_bytes:
                        st.image(fig_bytes, caption=f"Gráfico de Candlestick para {ativo_input} ({periodo_selecionado})")
                    else:
                        st.error("Não foi possível gerar o gráfico.")
            else:
                st.error(f"A análise para **{ativo_input}** falhou. Verifique o código do ativo e tente novamente.")

if __name__ == "__main__":
    main()

