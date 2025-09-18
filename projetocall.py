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

# Configurações dos indicadores técnicos
RSI_PERIOD = 9
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
MA_FAST = 21
MA_SLOW = 50
BBANDS_PERIOD = 20

# Lista de ativos populares para seleção rápida
ATIVOS_POPULARES = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "ELET3",
    "BBAS3", "WEGE3", "MGLU3", "RENT3", "PRIO3", "SUZB3"
]

# =================================================================
# 2. FUNÇÕES DE CÁLCULO E ANÁLISE
# =================================================================

def calcular_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona um conjunto de indicadores técnicos ao DataFrame.
    """
    # Pontos de Pivô Clássico
    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])

    # IFR (RSI)
    df.ta.rsi(length=RSI_PERIOD, append=True)

    # Bandas de Bollinger
    df.ta.bbands(length=BBANDS_PERIOD, append=True)

    # Média de Volume
    df['Volume Medio Mensal'] = df['Volume'].rolling(window=MA_FAST).mean()

    return df

@st.cache_data(ttl="15m") # Adiciona cache para otimizar a performance
def carregar_e_processar_dados(ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    Baixa os dados do ativo do yfinance e calcula todos os indicadores técnicos.
    A anotação @st.cache_data impede que os dados sejam baixados novamente a cada interação.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            st.error(f"Erro: Não foram encontrados dados para o ativo '{ticker}'. Verifique o código.")
            return None

        # yfinance pode retornar colunas com MultiIndex, isso normaliza
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = calcular_indicadores_tecnicos(df)
        return df

    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar os dados: {e}")
        return None

def gerar_relatorio_analise(df: pd.DataFrame, ticker_name: str) -> str:
    """
    Gera um relatório estruturado com análise técnica, recomendação e níveis de preço.
    """
    dados_atuais = df.iloc[-1]
    fechamento = dados_atuais['Close']
    ifr = dados_atuais[f'RSI_{RSI_PERIOD}']

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
    relatorio += f"**Data da Análise:** {df.index[-1].strftime('%d/%m/%Y')}\n\n"

    relatorio += "### Níveis de Preço Chave (Pivô Clássico)\n"
    relatorio += f"- **Ponto de Pivô (PP):** R$ {pp:.2f}\n"
    relatorio += f"- **Suporte 1 (S1):** R$ {s1:.2f}\n"
    relatorio += f"- **Resistência 1 (R1):** R$ {r1:.2f}\n"
    relatorio += f"- **Suporte 2 (S2):** R$ {s2:.2f}\n"
    relatorio += f"- **Resistência 2 (R2):** R$ {r2:.2f}\n\n"

    relatorio += f"### Análise de Indicadores\n"
    relatorio += f"O preço atual está entre o **Suporte 1 (R$ {s1:.2f})** e a **Resistência 1 (R$ {r1:.2f})**.\n"
    relatorio += f"- **Índice de Força Relativa (IFR):** {justificativa_ifr}\n"
    relatorio += f"- **Bandas de Bollinger:** O preço está "
    if fechamento > dados_atuais[f'BBU_{BBANDS_PERIOD}_2.0']:
        relatorio += "**acima da banda superior**, indicando alta volatilidade ou possível sobre-compra."
    elif fechamento < dados_atuais[f'BBL_{BBANDS_PERIOD}_2.0']:
        relatorio += "**abaixo da banda inferior**, indicando alta volatilidade ou possível sobre-venda."
    else:
        relatorio += f"**entre as bandas** (Superior: R$ {dados_atuais[f'BBU_{BBANDS_PERIOD}_2.0']:.2f}, Inferior: R$ {dados_atuais[f'BBL_{BBANDS_PERIOD}_2.0']:.2f})."
    relatorio += "\n\n"

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
    Gera o gráfico de candlestick e o retorna como um objeto de bytes.
    """
    # Níveis de Pivô (apenas o último ponto)
    pivot_levels = [
        df['PP'].iloc[-1], df['S1'].iloc[-1], df['R1'].iloc[-1],
        df['S2'].iloc[-1], df['R2'].iloc[-1]
    ]
    pivot_colors = ['blue', 'green', 'red', 'darkgreen', 'darkred']
    pivot_styles = [':', '--', '--', '-.', '-.']

    # Adiciona plots dos indicadores
    add_plots = [
        # Bandas de Bollinger
        mpf.make_addplot(df[[f'BBU_{BBANDS_PERIOD}_2.0', f'BBL_{BBANDS_PERIOD}_2.0']], color='purple', linestyle=':'),
        # IFR no painel inferior
        mpf.make_addplot(df[f'RSI_{RSI_PERIOD}'], panel=2, color='blue', ylabel=f'IFR({RSI_PERIOD})'),
        mpf.make_addplot([RSI_OVERBOUGHT] * len(df), panel=2, color='red', linestyle='--'),
        mpf.make_addplot([RSI_OVERSOLD] * len(df), panel=2, color='green', linestyle='--')
    ]

    # Configuração de estilo do gráfico com base no tema
    if theme == "Escuro":
        mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle=':', y_on_right=False)
    else: # Padrão é o tema Claro
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

    # Salva a figura em um buffer de bytes para exibir no Streamlit
    buf = BytesIO()
    mpf.plot(
        df,
        type='candle',
        style=s,
        title=f'Análise Técnica: {ativo_nome}',
        ylabel='Preço (R$)',
        volume=True,
        mav=(MA_FAST, MA_SLOW),
        addplot=add_plots,
        hlines=dict(hlines=pivot_levels, colors=pivot_colors, linestyle=pivot_styles, alpha=0.7),
        show_nontrading=False,
        figscale=1.5,
        panel_ratios=(3, 1), # Aumenta o espaço do gráfico de preço
        savefig=dict(fname=buf, format='png')
    )
    buf.seek(0)
    return buf.getvalue()

# =================================================================
# 3. INTERFACE DO STREAMLIT (UI)
# =================================================================

def main():
    st.set_page_config(page_title="Call Brava - Análise Técnica", layout="wide")
    st.title("📈 Call Brava")
    st.markdown("Análise técnica simplificada para ativos da B3, baseada em Pontos de Pivô, IFR e Bandas de Bollinger.")

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

        if df_processado is None or df_processado.empty:
            # A função carregar_e_processar_dados já exibe um erro
            return

        # --- EXIBIÇÃO DOS RESULTADOS ---
        st.header(f"Resultado da Análise para {ativo_input}")

        # Geração do relatório e do gráfico
        relatorio_texto = gerar_relatorio_analise(df_processado, ticker_yf)
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
                - **Bandas de Bollinger:** Linhas roxas pontilhadas.
                - **Níveis de Pivô:** R1/S1 (tracejadas), R2/S2 (traço-ponto).
                """.format(MA_FAST, MA_SLOW)
            )


if __name__ == "__main__":
    main()

