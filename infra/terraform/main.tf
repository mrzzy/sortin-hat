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
terraform {
  required_version = "~>1.2.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.31.0"
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

# GCS buckets
# store terraform state
resource "google_storage_bucket" "tf_state" {
  name     = "${local.project_id}-terraform-state"
  location = local.region

  lifecycle {
    prevent_destroy = true
  }
}


resource "google_storage_bucket" "buckets" {
  for_each = local.buckets
  name     = each.key
  location = local.region
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
resource "google_service_account" "k8s_node" {
  account_id   = "k8s-cluster-node"
  display_name = "Service account used by GKE 'main' K8s cluster worker nodes"
}
resource "google_container_node_pool" "infra" {
  name       = "support-infra"
  cluster    = google_container_cluster.main.id
  node_count = 1

  node_config {
    machine_type    = "e2-medium"
    service_account = google_service_account.k8s_node.email
    disk_size_gb    = 15
  }
}
