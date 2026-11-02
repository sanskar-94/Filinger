# EKS cluster for serving: a CPU node group for the API and a GPU node group (NVIDIA L4) for vLLM,
# IRSA roles scoped to the model and lake buckets, and ECR repositories for both images.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
  backend "s3" {
    bucket = "filings-lab-tfstate"
    key    = "eks/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" { region = var.region }

variable "region" { default = "us-east-1" }
variable "env" { default = "prod" }
variable "cluster_version" { default = "1.33" }

locals { name = "filings-lab-${var.env}" }

data "aws_availability_zones" "available" {}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name               = local.name
  cidr               = "10.40.0.0/16"
  azs                = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets    = ["10.40.0.0/19", "10.40.32.0/19", "10.40.64.0/19"]
  public_subnets     = ["10.40.96.0/22", "10.40.100.0/22", "10.40.104.0/22"]
  enable_nat_gateway = true
  single_nat_gateway = var.env != "prod"

  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name                             = local.name
  cluster_version                          = var.cluster_version
  vpc_id                                   = module.vpc.vpc_id
  subnet_ids                               = module.vpc.private_subnets
  cluster_endpoint_public_access           = true
  enable_irsa                              = true
  enable_cluster_creator_admin_permissions = true

  cluster_addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = {}
    eks-pod-identity-agent = {}
    aws-ebs-csi-driver     = {}
  }

  eks_managed_node_groups = {
    cpu = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 8
      desired_size   = 2
      labels         = { workload = "cpu" }
    }
    gpu = {
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = ["g6.xlarge"] # 1x NVIDIA L4, 24 GB: a 3B model in bf16 with room for KV cache
      min_size       = 1
      max_size       = 4
      desired_size   = 1
      capacity_type  = var.env == "prod" ? "ON_DEMAND" : "SPOT"
      labels         = { workload = "gpu" }
      taints = {
        gpu = { key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }
      }
      block_device_mappings = {
        xvda = { device_name = "/dev/xvda", ebs = { volume_size = 150, volume_type = "gp3" } }
      }
    }
  }
}

resource "aws_ecr_repository" "images" {
  for_each             = toset(["filings-api", "filings-vllm"])
  name                 = each.key
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

data "aws_iam_policy_document" "serving_read" {
  statement {
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::filings-lab-${var.env}",
      "arn:aws:s3:::filings-lab-${var.env}/registry/*",
      "arn:aws:s3:::filings-lab-${var.env}/adapters/*",
      "arn:aws:s3:::filings-lab-${var.env}/lake/gold/*",
    ]
  }
}

module "serving_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.44"

  role_name        = "filings-lab-serving"
  role_policy_arns = { read = aws_iam_policy.serving_read.arn }
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["filings-lab:filings-lab"]
    }
  }
}

resource "aws_iam_policy" "serving_read" {
  name   = "filings-lab-serving-read-${var.env}"
  policy = data.aws_iam_policy_document.serving_read.json
}

# GitHub Actions deploys through OIDC: no long lived AWS keys in the repository.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:sanskar-94/Filinger:environment:production"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "filings-lab-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

resource "aws_iam_role_policy_attachment" "github_ecr" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

output "cluster_name" { value = module.eks.cluster_name }
output "serving_role_arn" { value = module.serving_irsa.iam_role_arn }
output "github_deploy_role_arn" { value = aws_iam_role.github_deploy.arn }
