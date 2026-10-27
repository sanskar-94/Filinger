# EKS for serving

`terraform init && terraform apply -var env=prod` creates the VPC, the cluster with CPU and GPU node groups,
ECR repositories, the IRSA role the pods use to read models from S3, and the OIDC role GitHub Actions
assumes to deploy. Then install the cluster add ons once:

1. NVIDIA device plugin (or the GPU operator) so pods can request `nvidia.com/gpu`.
2. kube-prometheus-stack (Prometheus, Alertmanager, Grafana) in namespace `monitoring`.
3. prometheus-adapter with a rule exposing `flab_requests_per_second` to the HPA.
4. KEDA for queue based scaling of vLLM.
5. AWS Load Balancer Controller for the ALB ingress.

Apply the workloads with `kubectl apply -k infra/k8s/overlays/prod` (the deploy workflow does this).
