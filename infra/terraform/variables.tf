variable "project_name" {
  description = "Nome base do projeto."
  type        = string
  default     = "diagnostico-saude-mulher"
}

variable "aws_region" {
  description = "Regiao AWS."
  type        = string
  default     = "us-east-1"
}

variable "execution_role_arn" {
  description = "ARN da role de execucao do ECS."
  type        = string
}

variable "task_role_arn" {
  description = "ARN da role da tarefa ECS."
  type        = string
}
