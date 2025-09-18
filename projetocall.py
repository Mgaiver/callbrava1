import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
from datetime import date, timedelta
from io import BytesIO

# =================================================================
# 1. FUNÇÕES DE ANÁLISE E SINAIS
# =================================================================

def calcular_pivot_classico(df):
    """
    Calcula os Níveis de Pivô Clássico (PP, S1, R1, S2, R2).
    """
    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])
    return df

def gerar_mensagem_sinal(df, ticker_name):
    """
    Gera um relatório estruturado com análise técnica, recomendação e níveis de preço.
    """
    
    dados_atuais = df.iloc[-1]
    fechamento = dados_atuais['Close']
    ifr = dados_atuais['RSI_9']
    
    try:
        r1 = dados_atuais['R1']
        s1 = dados_atuais['S1']
        r2 = dados_atuais['R2']
        s2 = dados_atuais['S2']
        pp = dados_atuais['PP']
    except KeyError:
        return "Erro: Não foi possível calcular Suporte/Resistência."

    # Define a tolerância de proximidade (para determinar se o preço 'está perto' do S/R)
    tolerancia_perc = 0.015
    tolerancia_r1 = r1 * tolerancia_perc
    tolerancia_s1 = s1 * tolerancia_perc
    
    # --- 1. Determinação da Recomendação ---
    recomendacao_acao = ""
    nivel_entrada = 0.0
    justificativa_ifr = f"IFR(9) atual ({ifr:.2f}) indica condições neutras."

    # Lógica de COMPRA FORTE (Preço baixo E Sobre-Venda)
    if fechamento <= s1 + tolerancia_s1 and ifr < 35:
        recomendacao_acao = "**COMPRA AGRESSIVA / LONG**"
        nivel_entrada = s1
        justificativa_ifr = f"IFR(9) ({ifr:.2f}) está em zona de **Sobre-Venda (< 35)**. Confirma o ponto de reversão."
    
    # Lógica de VENDA FORTE (Preço alto E Sobre-Compra)
    elif fechamento >= r1 - tolerancia_r1 and ifr > 65:
        recomendacao_acao = "**VENDA AGRESSIVA / SHORT**"
        nivel_entrada = r1
        justificativa_ifr = f"IFR(9) ({ifr:.2f}) está em zona de **Sobre-Compra (> 65)**. Confirma o ponto de reversão."

    # Lógica de Compra Moderada (Atingiu Suporte)
    elif fechamento <= s1:
        recomendacao_acao = "**COMPRA MODERADA**"
        nivel_entrada = s1
        justificativa_ifr = f"IFR(9) ({ifr:.2f}) indica pressão de compra moderada."

    # Lógica de Venda Moderada (Atingiu Resistência)
    elif fechamento >= r1:
        recomendacao_acao = "**VENDA MODERADA**"
        nivel_entrada = r1
        justificativa_ifr = f"IFR(9) ({ifr:.2f}) indica pressão de venda moderada."
    
    # Lógica Neutra (Sem sinal claro)
    else:
        recomendacao_acao = "**NEUTRA / AGUARDAR**"
        nivel_entrada = 0.0
        justificativa_ifr = f"IFR(9) ({ifr:.2f}) está em zona neutra (35-65)."

    # --- 2. Montagem do Relatório Estruturado ---
    
    relatorio = f"## Relatório Técnico para {ticker_name.replace('.SA', '')}\n"
    relatorio += "=" * (len(relatorio) - 3) + "\n"
    relatorio += f"**Preço Atual:** R$ {fechamento:.2f}\n"
    relatorio += f"**Ponto de Pivô (PP):** R$ {pp:.2f}\n"
    relatorio += f"**Suporte Imediato (S1):** R$ {s1:.2f}\n"
    relatorio += f"**Resistência Imediata (R1):** R$ {r1:.2f}\n"
    relatorio += "\n"
    
    relatorio += "### Análise da Tendência e IFR\n"
    relatorio += f"Pela análise gráfica técnica, o ativo está atualmente entre o **Suporte 1 (R$ {s1:.2f})** e a **Resistência 1 (R$ {r1:.2f})**.\n"
    relatorio += f"- **Índice de Força Relativa (IFR9):** {justificativa_ifr}\n"
    relatorio += f"- **Distância até S1:** {abs(fechamento - s1) / s1 * 100:.2f}%\n"
    relatorio += f"- **Distância até R1:** {abs(fechamento - r1) / r1 * 100:.2f}%\n"

    relatorio += "\n"
    relatorio += "### Recomendação de Ação\n"
    
    if recomendacao_acao != "**NEUTRA / AGUARDAR**":
        acao_verbo = "COMPRAR" if "COMPRA" in recomendacao_acao else "VENDER"
        relatorio += f"Com base na combinação de preço atual e IFR, a recomendação é de **{recomendacao_acao}**.\n"
        relatorio += f"O ponto ideal de entrada para esta operação é no valor de **R$ {nivel_entrada:.2f}** ({acao_verbo} na zona de reversão).\n"
    else:
        relatorio += "**Aguardar:** Não há um sinal claro de reversão ou rompimento no momento. Recomenda-se esperar o preço se aproximar do Suporte 1 ou Resistência 1 para uma decisão mais assertiva."

    relatorio += "\n---\n"
    relatorio += f"**Próximos Níveis de Alerta:** Suporte 2 (R$ {s2:.2f}) e Resistência 2 (R$ {r2:.2f})."

    return relatorio

def plotar_grafico(df, ativo_nome):
    """
    Gera o gráfico e retorna-o como um objeto de bytes para o Streamlit.
    """
    
    # Níveis de Fibonacci
    max_price = df['High'].max()
    min_price = df['Low'].min()
    fib_levels = {
        '23.6%': min_price + 0.236 * (max_price - min_price),
        '38.2%': min_price + 0.382 * (max_price - min_price),
        '50.0%': min_price + 0.500 * (max_price - min_price),
        '61.8%': min_price + 0.618 * (max_price - min_price),
    }
    
    # NÍVEIS DE SUPORTE E RESISTÊNCIA DO PIVÔ (apenas o último ponto)
    pivot_levels = [df['PP'].iloc[-1], df['S1'].iloc[-1], df['R1'].iloc[-1], df['S2'].iloc[-1], df['R2'].iloc[-1]]
    
    # Configuração das linhas horizontais
    hlines = list(fib_levels.values()) + pivot_levels
    colors = ['red'] * len(fib_levels) + ['gray'] * len(pivot_levels)
    linestyles = ['--'] * len(fib_levels) + ['-'] * len(pivot_levels)

    # Configuração do IFR como painel
    add_plots = [
        mpf.make_addplot(df['RSI_9'], panel=2, color='blue', ylabel='IFR(9)'),
        mpf.make_addplot([70] * len(df), panel=2, color='red', linestyle='--'),
        mpf.make_addplot([30] * len(df), panel=2, color='green', linestyle='--')
    ]
    
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    
    # O Streamlit não exibe diretamente o objeto 'fig' do mplfinance, 
    # precisamos salvar a figura em um buffer de bytes e depois exibi-la.
    buf = BytesIO()
    
    mpf.plot(
        df, type='candle', style=s, title=f'Análise Técnica: {ativo_nome}', ylabel='Preço',
        volume=True, mav=(21, 50), addplot=add_plots,
        hlines=dict(hlines=hlines, linestyle=linestyles, colors=colors),
        show_nontrading=False, figscale=1.5,
        savef=buf # Salva no buffer de bytes
    )
    
    return buf.getvalue()

def main():
    st.set_page_config(page_title="Analisador Técnico B3", layout="wide")
    st.title("📈 Analisador Técnico de Ativos B3")
    st.markdown("Insira o código do ativo para análise de IFR e Níveis de Suporte/Resistência.")

    # --- ENTRADA DO USUÁRIO ---
    col1, col2 = st.columns([1, 2])
    
    # Widget de entrada de texto
    ativo_input = col1.text_input("Ativo (Ex: PETR4, VALE3)", "PETR4").upper().strip()
    
    # Widget de seleção de período (último 6 meses como padrão)
    periodo = col1.selectbox("Período de Análise:", 
                             ["Último 3 meses", "Último 6 meses", "Último 1 ano"], 
                             index=1)
    
    if periodo == "Último 3 meses":
        dias = 90
    elif periodo == "Último 6 meses":
        dias = 180
    else:
        dias = 365
        
    start_date = date.today() - timedelta(days=dias)
    end_date = date.today()
    
    if not ativo_input:
        st.warning("Por favor, insira um código de ativo.")
        return

    # --- PROCESSAMENTO ---
    TICKER_YFINANCE = ativo_input
    if not TICKER_YFINANCE.endswith('.SA'):
        TICKER_YFINANCE += '.SA'

    if col1.button("Executar Análise"):
        
        with st.spinner(f"Baixando e analisando dados de {ativo_input}..."):
            try:
                # A. OBTER DADOS
                df = yf.download(TICKER_YFINANCE, start=start_date, end=end_date, auto_adjust=False)
                
                if df.empty:
                    st.error(f"Erro: Não foi possível encontrar dados para **{ativo_input}**.")
                    return

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # B. CALCULAR INDICADORES
                df = calcular_pivot_classico(df)
                df.ta.rsi(length=9, append=True)
                df['Volume Medio Mensal'] = df['Volume'].rolling(window=21).mean()
                
                # C. GERAR SINAL
                sinal_resumo, mensagem, referencias = gerar_mensagem_sinal(df.copy(), TICKER_YFINANCE)
                
                # D. PLOTAGEM (adaptada para Streamlit)
                fig_bytes = plotar_grafico(df, ativo_input)
                
                # --- SAÍDA ---
                st.subheader(f"Resultado da Análise para {ativo_input}")
                
                # Exibir Sinal
                st.markdown(f"**SINAL PRINCIPAL:** {sinal_resumo}")
                st.markdown(mensagem)
                st.info(f"Níveis de Referência (PP, S/R): {referencias}")
                
                # Exibir Gráfico
                st.image(fig_bytes, caption=f"Gráfico de Candlestick com IFR, Fibonacci e Pivô ({periodo})")

            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {e}")

if __name__ == "__main__":
    main()
