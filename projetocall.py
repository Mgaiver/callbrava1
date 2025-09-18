import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from datetime import date, timedelta
from io import BytesIO
from typing import Dict, List, Optional

# =================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =================================================================

RSI_PERIOD = 9
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
MA_FAST = 21
MA_SLOW = 50

# Lista de ativos populares para seleção rápida
ATIVOS_POPULARES = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "ELET3",
    "B3SA3", "BBAS3", "RENT3", "WEGE3", "SUZB3", "GGBR4"
]

# =================================================================
# 2. FUNÇÕES DE PROCESSAMENTO E ANÁLISE
# =================================================================

@st.cache_data(ttl="15m")
def carregar_e_processar_dados(ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    Carrega os dados do Yahoo Finance e calcula os indicadores técnicos.
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

    # Garante que as colunas estão com nomes corretos
    df.columns = [col.capitalize() for col in df.columns]

    # Calcula Pontos de Pivô Clássico
    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])

    # Calcula IFR
    df.ta.rsi(length=RSI_PERIOD, append=True)

    # Média de Volume
    df['Volume Medio Mensal'] = df['Volume'].rolling(window=MA_FAST).mean()

    return df

def gerar_relatorio_analise(df: pd.DataFrame, ticker_name: str, analysis_timestamp: pd.Timestamp) -> str:
    """
    Gera um relatório estruturado com análise técnica, recomendação e níveis de preço.
    """
    dados_atuais = df.iloc[-1]
    fechamento = dados_atuais['Close']
    
    # --- Verificação de segurança para o IFR ---
    ifr_col = f'RSI_{RSI_PERIOD}'
    if ifr_col not in dados_atuais or pd.isna(dados_atuais[ifr_col]):
        return "Erro: Não foi possível calcular o IFR para o período solicitado. Tente um período mais longo."
    ifr = dados_atuais[ifr_col]

    # Extrai os níveis de pivô com segurança
    pp, r1, s1, r2, s2 = dados_atuais.get('PP'), dados_atuais.get('R1'), dados_atuais.get('S1'), dados_atuais.get('R2'), dados_atuais.get('S2')
    if any(v is None for v in [pp, r1, s1, r2, s2]):
        return "Erro: Não foi possível calcular os Pontos de Pivô."

    # Define a tolerância para proximidade do preço com suporte/resistência
    tolerancia_perc = 0.015
    tolerancia_r1 = r1 * tolerancia_perc
    tolerancia_s1 = s1 * tolerancia_perc

    # --- 1. Determinação da Recomendação ---
    recomendacao_acao = "**NEUTRA / AGUARDAR**"
    justificativa_ifr = f"IFR({RSI_PERIOD}) em {ifr:.2f} está em zona neutra ({RSI_OVERSOLD}-{RSI_OVERBOUGHT})."

    # Lógica de COMPRA (Preço baixo E Sobre-Venda)
    if fechamento <= s1 + tolerancia_s1 and ifr < RSI_OVERSOLD:
        recomendacao_acao = "**COMPRA AGRESSIVA / LONG**"
        justificativa_ifr = f"IFR({RSI_PERIOD}) em {ifr:.2f} está na zona de **Sobre-Venda (< {RSI_OVERSOLD})**, confirmando um possível ponto de reversão."
    elif fechamento <= s1:
        recomendacao_acao = "**COMPRA MODERADA**"
        justificativa_ifr = f"IFR({RSI_PERIOD}) em {ifr:.2f} indica pressão de compra moderada ao se aproximar do suporte."

    # Lógica de VENDA (Preço alto E Sobre-Compra)
    elif fechamento >= r1 - tolerancia_r1 and ifr > RSI_OVERBOUGHT:
        recomendacao_acao = "**VENDA AGRESSIVA / SHORT**"
        justificativa_ifr = f"IFR({RSI_PERIOD}) em {ifr:.2f} está na zona de **Sobre-Compra (> {RSI_OVERBOUGHT})**, confirmando um possível ponto de reversão."
    elif fechamento >= r1:
        recomendacao_acao = "**VENDA MODERADA**"
        justificativa_ifr = f"IFR({RSI_PERIOD}) em {ifr:.2f} indica pressão de venda moderada ao se aproximar da resistência."

    # --- 2. Montagem do Relatório Estruturado ---
    relatorio = f"## Análise Técnica para {ticker_name.replace('.SA', '')}\n\n"
    relatorio += f"**Preço de Fechamento:** R$ {fechamento:.2f}\n"
    relatorio += f"**Dados referentes ao fechamento de:** {df.index[-1].strftime('%d/%m/%Y')}\n"
    relatorio += f"**Análise gerada em:** {analysis_timestamp.strftime('%d/%m/%Y às %H:%M')}\n"
    relatorio += "**Fonte dos Dados:** Yahoo Finance\n\n"

    relatorio += "### Níveis de Preço Chave (Pivô Clássico)\n"
    relatorio += f"- **Ponto de Pivô (PP):** R$ {pp:.2f}\n"
    relatorio += f"- **Suporte 1 (S1):** R$ {s1:.2f}\n"
    relatorio += f"- **Resistência 1 (R1):** R$ {r1:.2f}\n"
    relatorio += f"- **Suporte 2 (S2):** R$ {s2:.2f}\n"
    relatorio += f"- **Resistência 2 (R2):** R$ {r2:.2f}\n\n"

    relatorio += f"### Análise de Indicadores\n"
    relatorio += f"O preço atual está entre o **Suporte 1 (R$ {s1:.2f})** e a **Resistência 1 (R$ {r1:.2f})**.\n"
    relatorio += f"- **Índice de Força Relativa (IFR):** {justificativa_ifr}\n\n"
    
    relatorio += "### Recomendação\n"
    if "NEUTRA" in recomendacao_acao:
        relatorio += "**Aguardar:** Não há um sinal claro de entrada no momento. Recomenda-se monitorar o ativo e esperar o preço se aproximar dos níveis de suporte/resistência com confirmação do IFR."
    else:
        acao_verbo = "COMPRAR" if "COMPRA" in recomendacao_acao else "VENDER"
        nivel_entrada = s1 if "COMPRA" in recomendacao_acao else r1
        relatorio += f"A recomendação é de **{recomendacao_acao}**. O preço está em uma zona de potencial reversão. O ponto de entrada ideal seria próximo de **R$ {nivel_entrada:.2f}** ({acao_verbo} no nível de suporte/resistência)."

    return relatorio

def plotar_grafico(df: pd.DataFrame, ativo_nome: str, theme: str = "Claro") -> bytes:
    """
    Gera o gráfico de candlestick e o retorna como um objeto de bytes, com estilo aprimorado.
    """
    # Níveis de Pivô (apenas o último ponto)
    pivots = df.iloc[-1]
    pivot_levels = [
        pivots.get('PP'), pivots.get('S1'), pivots.get('R1'),
        pivots.get('S2'), pivots.get('R2')
    ]
    # Filtra níveis nulos caso algum cálculo tenha falhado
    pivot_levels = [p for p in pivot_levels if p is not None]
    
    # Cores e estilos mais distintos para os níveis de pivô
    pivot_colors = ['#1f77b4', '#2ca02c', '#d62728', '#98df8a', '#ff9896'] # Azul, Verde, Vermelho, Verde Claro, Vermelho Claro
    pivot_styles = [':', '--', '--', '-.', '-.']

    # --- Adiciona plots dos indicadores (com verificação) ---
    add_plots = []

    # IFR no painel inferior
    ifr_col = f'RSI_{RSI_PERIOD}'
    if ifr_col in df.columns:
        add_plots.extend([
            mpf.make_addplot(df[ifr_col], panel=2, color='blue', ylabel=f'IFR({RSI_PERIOD})', width=0.8),
            mpf.make_addplot([RSI_OVERBOUGHT] * len(df), panel=2, color='red', linestyle='--', width=1.2),
            mpf.make_addplot([RSI_OVERSOLD] * len(df), panel=2, color='green', linestyle='--', width=1.2)
        ])

    # Configuração de estilo do gráfico com base no tema
    if theme == "Escuro":
        # Tema escuro com cores de alto contraste
        mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True)
        s = mpf.make_mpf_style(
            base_mpf_style='nightclouds', marketcolors=mc, gridstyle=':', y_on_right=False,
            rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
        )
        watermark_color = 'white'
    else: # Padrão é o tema Claro
        # Tema claro e limpo, baseado no estilo do Yahoo Finance
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(
            base_mpf_style='yahoo', marketcolors=mc, gridstyle='--', y_on_right=False
        )
        watermark_color = 'gray'

    # Salva a figura em um buffer de bytes para exibir no Streamlit
    buf = BytesIO()
    mpf.plot(
        df,
        type='candle',
        style=s,
        title=f"\nAnálise Técnica: {ativo_nome}", # Adiciona espaço no topo
        ylabel='Preço (R$)',
        volume=True,
        ylabel_lower='Volume',
        mav=(MA_FAST, MA_SLOW),
        addplot=add_plots,
        hlines=dict(hlines=pivot_levels, colors=pivot_colors, linestyle=pivot_styles, alpha=0.8, linewidths=1.2),
        show_nontrading=False,
        figscale=1.8, # Gráfico maior e mais nítido
        panel_ratios=(4, 1), # Mais espaço para o gráfico de preço
        watermark=dict(text="Brava", color=watermark_color, alpha=0.3, fontsize=12),
        savefig=dict(fname=buf, format='png', bbox_inches='tight') # bbox_inches para evitar cortes
    )
    buf.seek(0)
    return buf.getvalue()

# =================================================================
# 3. INTERFACE DO STREAMLIT (UI)
# =================================================================

def main():
    st.set_page_config(page_title="Call Brava - Análise Técnica", layout="wide")
    st.title("📈 Call Brava")
    st.markdown("Análise técnica simplificada para ativos da B3, baseada em Pontos de Pivô e IFR.")

    # --- ENTRADA DO USUÁRIO NA BARRA LATERAL ---
    with st.sidebar:
        st.header("Configurações da Análise")

        # Seleção de ativo (lista + campo customizado)
        selecao_ativo = st.selectbox("Selecione um ativo popular:", ATIVOS_POPULARES, index=0)
        ativo_customizado = st.text_input("Ou digite um código (Ex: VIIA3):").upper().strip()
        
        ativo_input = ativo_customizado if ativo_customizado else selecao_ativo

        # Seleção de período
        periodo_map: Dict[str, int] = {
            "Últimos 3 meses": 90,
            "Últimos 6 meses": 180,
            "Último 1 ano": 365,
            "Últimos 2 anos": 730
        }
        periodo_selecionado = st.selectbox(
            "Período de Análise:",
            list(periodo_map.keys()),
            index=1
        )
        dias = periodo_map[periodo_selecionado]
        start_date = date.today() - timedelta(days=dias)

        # Seletor de tema para o gráfico
        tema_grafico = st.radio("Tema do Gráfico:", ["Claro", "Escuro"])

        # Botão para executar
        if not st.button("Executar Análise", type="primary", use_container_width=True):
            st.info("Selecione um ativo e clique em 'Executar Análise' para começar.")
            return

    # --- LÓGICA PRINCIPAL DE PROCESSAMENTO ---
    if not ativo_input:
        st.warning("Por favor, selecione ou digite um código de ativo.")
        return

    # Adiciona o sufixo .SA, padrão para ativos brasileiros no yfinance
    ticker_yf = f"{ativo_input}.SA" if not ativo_input.endswith('.SA') else ativo_input

    with st.spinner(f"Buscando e analisando dados de {ativo_input}..."):
        df_processado = carregar_e_processar_dados(ticker_yf, start_date, date.today())
        timestamp_geracao = pd.Timestamp.now()

        if df_processado is None or df_processado.empty:
            # A função carregar_e_processar_dados já exibe um erro
            return

        # --- EXIBIÇÃO DOS RESULTADOS ---
        st.header(f"Resultado da Análise para {ativo_input}")

        # Geração do relatório e do gráfico
        relatorio_texto = gerar_relatorio_analise(df_processado, ticker_yf, timestamp_geracao)
        fig_bytes = plotar_grafico(df_processado, ativo_input, theme=tema_grafico)

        # Organiza a saída em abas
        tab_relatorio, tab_grafico = st.tabs(["📄 Relatório de Análise", "📊 Gráfico Técnico"])

        with tab_relatorio:
            st.markdown(relatorio_texto)

        with tab_grafico:
            st.image(fig_bytes, caption=f"Gráfico de Candlestick para {ativo_input} ({periodo_selecionado})")
            st.markdown(
                """
                **Legenda do Gráfico:**
                - **Médias Móveis:** Laranja (curta, {} dias), Roxa (longa, {} dias).
                - **Níveis de Pivô:** R1/S1 (tracejadas), R2/S2 (traço-ponto).
                """.format(MA_FAST, MA_SLOW)
            )

if __name__ == "__main__":
    main()

