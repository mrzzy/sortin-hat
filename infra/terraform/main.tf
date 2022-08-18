#
# sortin-hat
# Terraform Infrastructure
#

locals {
  project_id = "sss-sortin-hat"
  region     = "asia-southeast1" # Google's SG, Jurong West datacenter
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
# raw data source files
resource "google_storage_bucket" "raw" {
  name     = "${local.project_id}-raw"
  location = local.region
}
# processed datasets for training ML models
resource "google_storage_bucket" "datasets" {
  name     = "${local.project_id}-datasets"
  location = local.region
}
# trained ML model artifacts
resource "google_storage_bucket" "models" {
  name     = "${local.project_id}-models"
  location = local.region
}

# GKE K8s Cluster
resource "google_container_cluster" "main" {
  name     = "main"
  location = local.region

  # manage node pools separately from the cluster: delete default node pool.
  remove_default_node_pool = true
  initial_node_count       = 1
}
