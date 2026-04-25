output "ecr_repository_url" {
  value       = aws_ecr_repository.app.repository_url
  description = "URL do repositorio ECR."
}

output "artifacts_bucket" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "Bucket para artefatos e logs."
}

output "ecs_cluster_name" {
  value       = aws_ecs_cluster.main.name
  description = "Nome do cluster ECS."
}
