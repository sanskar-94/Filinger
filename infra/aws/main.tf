# Core AWS resources for the lakehouse: versioned, encrypted S3 bucket with lifecycle rules,
# and an IAM role that Databricks / EMR assume to read and write it.
terraform {
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}

provider "aws" { region = var.region }

variable "region" { default = "us-east-1" }
variable "env" { default = "dev" }

resource "aws_s3_bucket" "lake" {
  bucket = "filings-lab-${var.env}"
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    id     = "raw-to-infrequent-access"
    status = "Enabled"
    filter { prefix = "lake/bronze/raw/" }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com", "elasticmapreduce.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pipeline" {
  name               = "filings-lab-pipeline-${var.env}"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "lake_rw" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.lake.arn, "${aws_s3_bucket.lake.arn}/*"]
  }
}

resource "aws_iam_role_policy" "lake_rw" {
  role   = aws_iam_role.pipeline.id
  policy = data.aws_iam_policy_document.lake_rw.json
}

resource "aws_iam_instance_profile" "pipeline" {
  name = "filings-lab-${var.env}"
  role = aws_iam_role.pipeline.name
}

output "bucket" { value = aws_s3_bucket.lake.bucket }
output "instance_profile_arn" { value = aws_iam_instance_profile.pipeline.arn }
