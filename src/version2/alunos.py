
#   Só as variáveis principais precisam ser informadas: o restante assume o valor típico (mediana) do treino.
#   Os cliques entram na escala ORIGINAL (o previsao.py aplica o log).
#
#   vle_total_cliques       : cliques no ambiente virtual até o dia do corte
#   vle_dias_ativos         : dias distintos com acesso
#   vle_cliques_ult_14d     : cliques nos 14 dias anteriores ao corte
#   vle_dias_sem_acesso     : dias desde o último acesso até o corte
#   aval_n_entregues        : avaliações entregues | aval_n_devidas = avaliações com prazo até o corte
#   aval_score_medio        : nota média (0 a 100)  | aval_score_ponderado_soma = soma das notas x peso
#   aval_taxa_entrega       : entregues / devidas (0 a 1)
#   aval_n_atrasadas        : entregas feitas depois do prazo
#   num_tentativas_anteriores : quantas vezes já tentou este módulo
#   codigo_modulo           : AAA, BBB, CCC, DDD, EEE, FFF ou GGG

ALUNOS = {
    # Engajado e em dia com as entregas
    "Baixo risco": {
        "vle_total_cliques": 1500,
        "vle_dias_ativos": 60,
        "vle_cliques_ult_14d": 120,
        "vle_dias_sem_acesso": 1,
        "aval_n_entregues": 3,
        "aval_n_devidas": 3,
        "aval_taxa_entrega": 1.0,
        "aval_score_medio": 88,
        "aval_score_ponderado_soma": 55,
        "aval_n_atrasadas": 0,
        "aval_sem_nota": 0,
        "nunca_acessou_vle": 0,
        "num_tentativas_anteriores": 0,
        "codigo_modulo": "GGG",
    },

    # Acesso irregular, entregas atrasadas e nota na média
    "Risco intermediario": {
        "vle_total_cliques": 350,
        "vle_dias_ativos": 22,
        "vle_cliques_ult_14d": 15,
        "vle_dias_sem_acesso": 9,
        "aval_n_entregues": 2,
        "aval_n_devidas": 3,
        "aval_taxa_entrega": 0.67,
        "aval_score_medio": 68,
        "aval_score_ponderado_soma": 20,
        "aval_n_atrasadas": 1,
        "aval_sem_nota": 0,
        "nunca_acessou_vle": 0,
        "num_tentativas_anteriores": 1,
        "codigo_modulo": "FFF",
    },

    # Quase sumiu do ambiente e não entregou as avaliações
    "Alto risco": {
        "vle_total_cliques": 40,
        "vle_dias_ativos": 4,
        "vle_cliques_ult_14d": 0,
        "vle_dias_sem_acesso": 60,
        "aval_n_entregues": 0,
        "aval_n_devidas": 3,
        "aval_taxa_entrega": 0.0,
        "aval_score_medio": None,       # sem nota (não entregou nada)
        "aval_score_ponderado_soma": 0,
        "aval_n_atrasadas": 0,
        "aval_sem_nota": 1,
        "nunca_acessou_vle": 0,
        "num_tentativas_anteriores": 2,
        "codigo_modulo": "CCC",
    },
}
