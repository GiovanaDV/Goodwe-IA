import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime, timedelta

# ===============================
# 1) Carregar e Preparar Dados
# ===============================
caminho_pasta = "C:\\Users\\giova\\OneDrive\\Desktop\\FIAP\\dados IA GOODWE"
arquivos = glob.glob(os.path.join(caminho_pasta, "*.xls")) + glob.glob(os.path.join(caminho_pasta, "*.xlsx"))

print("🔍 SISTEMA DETETIVE ENERGÉTICO")
print("=" * 40)
print(f"📂 Arquivos encontrados: {len(arquivos)}")

lista_dfs = []

for arq in arquivos:
    try:
        if arq.endswith(".xlsx"):
            df_temp = pd.read_excel(arq, header=2, engine="openpyxl")
        else:
            df_temp = pd.read_excel(arq, header=2, engine="xlrd")

        df_temp.columns = df_temp.columns.str.strip().str.lower()

        if "time" in df_temp.columns:
            df_temp["time"] = pd.to_datetime(df_temp["time"], errors="coerce", dayfirst=True)

        for col in df_temp.columns:
            if "load" in col:
                df_temp.rename(columns={col: "load"}, inplace=True)

        df_temp = df_temp.dropna(subset=["time", "load"])
        df_temp["load"] = pd.to_numeric(df_temp["load"], errors="coerce")
        df_temp = df_temp[df_temp["load"] >= 0]  # Remover valores negativos

        lista_dfs.append(df_temp)
        print(f"✅ {os.path.basename(arq)}: {len(df_temp)} registros")

    except Exception as e:
        print(f"⚠️ Erro em {arq}: {e}")

df = pd.concat(lista_dfs, ignore_index=True)
df = df.sort_values('time').reset_index(drop=True)

print(f"\n📊 Dataset consolidado: {len(df)} registros de {df['time'].dt.date.nunique()} dias")


# ===============================
# 2) ANÁLISE DETETIVE: Descobrir Padrões
# ===============================
class DetetiveEnergetico:
    def __init__(self, dados):
        self.dados = dados
        self.perfil_casa = {}

    def analisar_casa(self):
        """Analisa os dados para entender o perfil da casa"""
        print("\n🕵️ ANÁLISE DETETIVE DA CASA")
        print("-" * 30)

        # 1. Consumo base (equipamentos sempre ligados)
        consumo_minimo = self.dados['load'].quantile(0.05)  # 5% menores valores
        consumo_base = self.dados['load'].quantile(0.10)  # Provável consumo base

        # 2. Diferentes tipos de consumo
        consumo_baixo = self.dados['load'].quantile(0.25)  # Baixa atividade
        consumo_medio = self.dados['load'].quantile(0.50)  # Atividade normal
        consumo_alto = self.dados['load'].quantile(0.75)  # Alta atividade
        consumo_pico = self.dados['load'].quantile(0.95)  # Picos de uso

        # 3. Análise temporal
        self.dados['hora'] = self.dados['time'].dt.hour
        self.dados['dia_semana'] = self.dados['time'].dt.dayofweek
        self.dados['fim_semana'] = (self.dados['dia_semana'] >= 5).astype(int)

        # 4. Classificar períodos do dia
        def classificar_periodo(hora):
            if 6 <= hora < 12:
                return "manha"
            elif 12 <= hora < 18:
                return "tarde"
            elif 18 <= hora < 23:
                return "noite"
            else:
                return "madrugada"

        self.dados['periodo'] = self.dados['hora'].apply(classificar_periodo)

        # 5. Perfil por período
        perfil_temporal = self.dados.groupby('periodo')['load'].agg(['mean', 'std', 'min', 'max'])

        # Armazenar descobertas
        self.perfil_casa = {
            'consumo_minimo': consumo_minimo,
            'consumo_base': consumo_base,
            'consumo_baixo': consumo_baixo,
            'consumo_medio': consumo_medio,
            'consumo_alto': consumo_alto,
            'consumo_pico': consumo_pico,
            'perfil_temporal': perfil_temporal,
            'standby_estimado': (consumo_baixo - consumo_base) * 0.6  # 60% da diferença pode ser standby
        }

        # Relatório das descobertas
        print(f"🏠 Consumo base da casa: {consumo_base:.1f}W (geladeira, roteador, etc)")
        print(f"😴 Consumo em baixa atividade: {consumo_baixo:.1f}W")
        print(f"⚡ Standby estimado disponível: {self.perfil_casa['standby_estimado']:.1f}W")
        print(f"📈 Consumo médio: {consumo_medio:.1f}W")
        print(f"🔥 Picos de consumo: {consumo_pico:.1f}W")

        print(f"\n⏰ Perfil por período:")
        for periodo in perfil_temporal.index:
            media = perfil_temporal.loc[periodo, 'mean']
            print(f"  {periodo.capitalize()}: {media:.1f}W em média")

        return self.perfil_casa

    def detectar_estados_casa(self):
        """Detecta diferentes estados da casa baseado no consumo"""

        # Definir estados baseado nos thresholds descobertos
        def classificar_estado(row):
            consumo = row['load']
            hora = row['hora']
            periodo = row['periodo']

            # Regras de classificação de estado
            if consumo <= self.perfil_casa['consumo_base'] * 1.1:
                return "casa_vazia"
            elif consumo <= self.perfil_casa['consumo_baixo']:
                if periodo == "madrugada":
                    return "dormindo"
                else:
                    return "atividade_baixa"
            elif consumo <= self.perfil_casa['consumo_medio']:
                return "atividade_normal"
            else:
                return "atividade_alta"

        self.dados['estado_casa'] = self.dados.apply(classificar_estado, axis=1)

        # Estatísticas dos estados
        distribuicao_estados = self.dados['estado_casa'].value_counts(normalize=True) * 100

        print(f"\n🏠 ESTADOS DETECTADOS DA CASA:")
        for estado, percentual in distribuicao_estados.items():
            print(f"  {estado.replace('_', ' ').title()}: {percentual:.1f}% do tempo")

        return distribuicao_estados


# ===============================
# 3) SISTEMA INTELIGENTE DE DECISÃO
# ===============================
class SistemaDecisaoIA:
    def __init__(self, perfil_casa):
        self.perfil = perfil_casa
        self.regras_seguranca = self._criar_regras_seguranca()

    def _criar_regras_seguranca(self):
        """Define regras de segurança baseadas em comportamento humano típico"""
        return {
            'horarios_conservadores': [7, 8, 18, 19, 20, 21],  # Rotina matinal e noturna
            'nunca_desligar_abaixo': self.perfil['consumo_base'],  # Nunca abaixo do consumo base
            'madrugada_segura': [0, 1, 2, 3, 4, 5, 23],  # Horários seguros para otimização
            'economia_maxima_segura': self.perfil['standby_estimado'] * 0.8,  # Máx 80% do standby estimado
            'tempo_minimo_estavel': 3  # Aguardar 3 medições estáveis antes de desligar
        }

    def decidir_acao(self, dados_momento):
        """Decide se deve desligar standby neste momento"""

        consumo = dados_momento['load']
        hora = dados_momento['hora']
        estado = dados_momento['estado_casa']
        periodo = dados_momento['periodo']
        dia_semana = dados_momento['dia_semana']

        # Inicializar decisão
        decisao = {
            'acao': 'manter',
            'confianca': 0.0,
            'motivo': '',
            'economia_estimada': 0,
            'seguro_desligar': False
        }

        # REGRA 1: Nunca desligar se consumo muito baixo (equipamentos essenciais)
        if consumo <= self.regras_seguranca['nunca_desligar_abaixo']:
            decisao.update({
                'acao': 'manter',
                'motivo': 'Consumo muito baixo - proteção equipamentos essenciais',
                'confianca': 0.95
            })
            return decisao

        # REGRA 2: Estados favoráveis para desligar standby
        estados_favoraveis = ['casa_vazia', 'dormindo', 'atividade_baixa']

        if estado in estados_favoraveis:
            confianca_base = 0.7
            economia_possivel = min(
                (consumo - self.perfil['consumo_base']) * 0.3,  # 30% da diferença
                self.regras_seguranca['economia_maxima_segura']
            )

            # REGRA 3: Horários seguros aumentam confiança
            if hora in self.regras_seguranca['madrugada_segura']:
                confianca_base += 0.2

            # REGRA 4: Fins de semana = padrões diferentes
            if dia_semana >= 5:  # Fim de semana
                if estado == 'dormindo' or (estado == 'atividade_baixa' and hora in [9, 10, 11, 14, 15, 16]):
                    confianca_base += 0.1

            # REGRA 5: Dias úteis durante horário comercial
            if dia_semana < 5 and 9 <= hora <= 17:
                if estado in ['casa_vazia', 'atividade_baixa']:
                    confianca_base += 0.15

            # REGRA 6: Evitar horários de rotina intensa
            if hora in self.regras_seguranca['horarios_conservadores']:
                confianca_base -= 0.2

            # Decidir baseado na confiança
            if confianca_base >= 0.6 and economia_possivel > 5:  # Mín 5W de economia
                decisao.update({
                    'acao': 'desligar_standby',
                    'confianca': min(confianca_base, 0.95),
                    'motivo': f'Estado: {estado}, Período: {periodo}, Economia segura detectada',
                    'economia_estimada': economia_possivel,
                    'seguro_desligar': True
                })
            else:
                decisao.update({
                    'acao': 'manter',
                    'motivo': f'Baixa confiança ({confianca_base:.2f}) ou economia insuficiente',
                    'confianca': confianca_base
                })

        else:  # Estados não favoráveis
            decisao.update({
                'acao': 'manter',
                'motivo': f'Estado não favorável: {estado}',
                'confianca': 0.8
            })

        return decisao


# ===============================
# 4) EXECUTAR ANÁLISE COMPLETA
# ===============================

# Criar detetive e analisar
detetive = DetetiveEnergetico(df)
perfil = detetive.analisar_casa()
distribuicao_estados = detetive.detectar_estados_casa()

# Criar sistema de decisão
sistema_ia = SistemaDecisaoIA(perfil)

# ===============================
# 5) SIMULAÇÃO EM UM DIA COMPLETO
# ===============================

# Escolher um dia para testar
dia_teste = df['time'].dt.date.unique()[0]  # Último dia
df_dia = df[df['time'].dt.date == dia_teste].copy()

print(f"\n🎯 SIMULAÇÃO DO SISTEMA IA - {dia_teste}")
print("=" * 50)

# Aplicar decisões para cada momento do dia
decisoes_dia = []
consumo_otimizado = []

for idx, row in df_dia.iterrows():
    decisao = sistema_ia.decidir_acao(row)
    decisoes_dia.append(decisao)

    # Aplicar economia se decisão foi desligar
    if decisao['acao'] == 'desligar_standby':
        consumo_novo = max(perfil['consumo_base'], row['load'] - decisao['economia_estimada'])
    else:
        consumo_novo = row['load']

    consumo_otimizado.append(consumo_novo)

df_dia['decisao_ia'] = [d['acao'] for d in decisoes_dia]
df_dia['confianca'] = [d['confianca'] for d in decisoes_dia]
df_dia['economia_estimada'] = [d['economia_estimada'] for d in decisoes_dia]
df_dia['motivo'] = [d['motivo'] for d in decisoes_dia]
df_dia['consumo_otimizado'] = consumo_otimizado

# ===============================
# 6) VISUALIZAÇÃO INTELIGENTE
# ===============================

plt.figure(figsize=(16, 12))

# Gráfico 1: Consumo Real vs Otimizado
plt.subplot(3, 1, 1)
plt.plot(df_dia['time'], df_dia['load'], label='Consumo Real', color='red', linewidth=2)
plt.plot(df_dia['time'], df_dia['consumo_otimizado'], label='Consumo Otimizado pela IA', color='green', linewidth=2)

# Marcar momentos de intervenção da IA
standby_desligado = df_dia[df_dia['decisao_ia'] == 'desligar_standby']
plt.scatter(standby_desligado['time'], standby_desligado['load'],
            color='blue', s=60, alpha=0.8, label='IA desligou standby', zorder=5)

# Linhas de referência
plt.axhline(y=perfil['consumo_base'], color='orange', linestyle='--', alpha=0.7,
            label=f'Consumo Base ({perfil["consumo_base"]:.1f}W)')
plt.axhline(y=perfil['consumo_baixo'], color='gray', linestyle=':', alpha=0.7,
            label=f'Threshold Baixo Consumo ({perfil["consumo_baixo"]:.1f}W)')

plt.title(f'Sistema IA Detetive Energético - {dia_teste}', fontsize=14, fontweight='bold')
plt.ylabel('Potência (W)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Estados da Casa
plt.subplot(3, 1, 2)
cores_estados = {
    'casa_vazia': 'lightblue',
    'dormindo': 'purple',
    'atividade_baixa': 'yellow',
    'atividade_normal': 'orange',
    'atividade_alta': 'red'
}

for estado in df_dia['estado_casa'].unique():
    mask = df_dia['estado_casa'] == estado
    plt.scatter(df_dia[mask]['time'], [estado] * sum(mask),
                color=cores_estados.get(estado, 'gray'), alpha=0.7, s=30)

plt.title('Estados Detectados da Casa')
plt.ylabel('Estado')
plt.grid(True, alpha=0.3)

# Gráfico 3: Confiança das Decisões
plt.subplot(3, 1, 3)
plt.plot(df_dia['time'], df_dia['confianca'], color='purple', linewidth=2)
plt.fill_between(df_dia['time'], df_dia['confianca'], alpha=0.3, color='purple')
plt.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold de Ação (60%)')

plt.title('Nível de Confiança das Decisões da IA')
plt.ylabel('Confiança')
plt.xlabel('Horário')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# 7) RELATÓRIO FINAL DETALHADO
# ===============================

# Calcular métricas
total_intervencoes = len(standby_desligado)
economia_total = df_dia['economia_estimada'].sum()
consumo_original = df_dia['load'].sum()
consumo_otimizado_total = df_dia['consumo_otimizado'].sum()
percentual_economia = (economia_total / consumo_original) * 100
tempo_otimizacao = (total_intervencoes / len(df_dia)) * 100
confianca_media = df_dia['confianca'].mean()

print(f"\n📊 RELATÓRIO FINAL - SISTEMA DETETIVE ENERGÉTICO")
print("=" * 60)
print(f"📅 Dia analisado: {dia_teste}")
print(f"⏱️  Total de medições: {len(df_dia)}")
print()
print("🔍 DESCOBERTAS DO DETETIVE:")
print(f"   🏠 Consumo base da casa: {perfil['consumo_base']:.1f}W")
print(f"   💤 Standby estimado disponível: {perfil['standby_estimado']:.1f}W")
print(f"   📊 Consumo médio do dia: {df_dia['load'].mean():.1f}W")
print()
print("🤖 PERFORMANCE DA IA:")
print(f"   ⚡ Intervenções realizadas: {total_intervencoes}")
print(f"   ⏰ Tempo sob otimização: {tempo_otimizacao:.1f}% do dia")
print(f"   🎯 Confiança média: {confianca_media:.1%}")
print()
print("💰 RESULTADOS ECONÔMICOS:")
print(f"   📈 Consumo original: {consumo_original:.2f} Wh")
print(f"   📉 Consumo otimizado: {consumo_otimizado_total:.2f} Wh")
print(f"   💵 Economia total: {economia_total:.2f} Wh ({percentual_economia:.2f}%)")
print()
print("📈 ANÁLISE POR ESTADO:")
for estado in df_dia['estado_casa'].unique():
    mask_estado = df_dia['estado_casa'] == estado
    intervencoes_estado = df_dia[mask_estado]['decisao_ia'].value_counts()
    total_estado = len(df_dia[mask_estado])

    if 'desligar_standby' in intervencoes_estado:
        perc_otim = (intervencoes_estado['desligar_standby'] / total_estado) * 100
        print(f"   {estado.replace('_', ' ').title()}: {perc_otim:.1f}% otimizado")

print(f"\n🎉 Sistema funcionando! A IA está tomando decisões inteligentes baseada nos padrões detectados!")