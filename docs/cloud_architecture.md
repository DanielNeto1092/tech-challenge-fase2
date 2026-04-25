# Arquitetura de Nuvem Proposta

## Objetivo

Este documento descreve uma arquitetura alvo para implantacao do sistema em ambiente real, considerando escalabilidade, observabilidade e separacao de responsabilidades.

## Arquitetura sugerida

```text
Usuario -> Load Balancer -> App Streamlit/API containerizada
                                 |-> Servico de execucao de experimentos
                                 |-> Persistencia de artefatos
                                 |-> Logs e monitoramento
```

## Componentes recomendados

- `ECR`: armazenamento da imagem Docker.
- `ECS Fargate` ou `Kubernetes`: execucao containerizada sem dependencia de servidor fixo.
- `Application Load Balancer`: distribuicao de trafego.
- `S3`: armazenamento de artefatos, relatorios, modelos e respostas da LLM.
- `CloudWatch`: centralizacao de logs, metricas e alertas.
- `Lambda` ou worker assíncrono: execucao futura de cargas pesadas em background.

## Escalabilidade

- A interface web pode escalar horizontalmente com multiplas replicas.
- A otimizacao genetica pode ser desacoplada da interface e enviada para jobs assíncronos.
- O armazenamento de artefatos deve ser externo ao container.
- O uso de fila permite evitar bloquear a interface em demandas intensivas.

## Monitoramento recomendado

- taxa de erro das requisicoes;
- tempo medio de resposta;
- tempo por experimento genetico;
- quantidade de chamadas a LLM;
- tamanho do historico de respostas;
- saturacao de CPU e memoria dos containers.

## IaC

Foi incluida uma estrutura inicial em [infra/terraform](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/infra/terraform) com:

- `main.tf`
- `variables.tf`
- `outputs.tf`

Essa base prepara o projeto para evolucao futura em AWS usando ECS Fargate, ECR, S3 e CloudWatch.
