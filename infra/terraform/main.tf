#
# sortin-hat
# Terraform Infrastructure
#

locals {
  project_id = "sss-sortin-hat"
  # Google's SG, Jurong West datacenter
  region = "asia-southeast1"

  # name of the GCS buckets to create
  buckets = toset([for suffix in [
    # raw, unprocessed source data files
    "raw-data",
    # processed datasets for training ML models
    "datasets",
    # trained ML model artifacts
    "models"
  ] : "${local.project_id}-${suffix}"])
}

# Terraform & Provider config
terraform {
  required_version = "~>1.2.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.31.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.13.0"
    }
  }
  backend "gcs" {
    bucket = "sss-sortin-hat-terraform-state"
    prefix = "terraform/state"
  }
}
provider "google" {
  project = local.project_id
  region  = local.region
}

# fetch access token using Google Application Default credentials to authenticate
# terraform's kubernetes provider when applying resources.
data "google_client_config" "default" {}
provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth.0.cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}

# GCS buckets
# store terraform state
resource "google_storage_bucket" "tf_state" {
  name     = "${local.project_id}-terraform-state"
  location = local.region

  lifecycle {
    prevent_destroy = true
  }
}

# Google Cloud Storage
resource "google_storage_bucket" "buckets" {
  for_each = local.buckets
  name     = each.key
  location = local.region
}

# GCP IAM
# service account to allow ML pipeline containers to access GCS Buckets
resource "google_service_account" "pipeline" {
  account_id   = "ml-pipeline"
  display_name = "Service account used by ML pipeline containers to access worker nodes"
}
resource "google_storage_bucket_iam_member" "allow_k8s" {
  for_each = local.buckets
  bucket   = each.key
  role     = "roles/storage.objectAdmin" # allow CRUD operations on object
  member   = "serviceAccount:${google_service_account.pipeline.email}"
}
# ml pipeline service account as k8s secret
resource "google_service_account_key" "pipeline" {
  service_account_id = google_service_account.pipeline.id
}
resource "kubernetes_secret_v1" "pipeline_svc_acc_key" {
  metadata {
    name = "ml-pipeline-gcp-service-account"
    labels = {
      "app.kubernetes.io/part-of"    = "sortin-hat"
      "app.kubernetes.io/component"  = "ml-pipeline"
      "app.kubernetes.io/managed-by" = "terraform"
    }

  }
  data = {
    "private-key" = base64decode(google_service_account_key.pipeline.private_key)
  }
}

# service account for GKE worker nodes
resource "google_service_account" "k8s_node" {
  account_id   = "k8s-cluster-node"
  display_name = "Service account used by GKE 'main' K8s cluster worker nodes"
}

# GKE K8s Cluster
resource "google_container_cluster" "main" {
  name = "main"
  # deploy a zonal cluster as regional cluster will spin up 1 node per zone (3)
  # which is too much for our requirements
  location = "${local.region}-c"

  # manage node pools separately from the cluster: delete default node pool.
  remove_default_node_pool = true
  initial_node_count       = 1
}
# node pool for running long-running support infrastructure (ie. Airflow, MLFlow)
resource "google_container_node_pool" "infra" {
  name       = "support-infra"
  cluster    = google_container_cluster.main.id
  node_count = 1

  node_config {
    machine_type    = "e2-standard-4" # 2vCPU, 4GB RAM
    service_account = google_service_account.k8s_node.email
    disk_size_gb    = 30
  }
}
