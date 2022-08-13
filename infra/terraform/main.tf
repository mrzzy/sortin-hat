#
# sortin-hat
# Terraform Infrastructure
#

locals {
  project_id = "sss-sortin-hat"
  # TODO(mrzzy): merge with region
  gcs_location = "ASIA-SOUTHEAST1" # Singapore
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
  region  = "asia-southeast1" # Google's SG, Jurong West datacenter
}

# GCS buckets
# store terraform state
resource "google_storage_bucket" "tf_state" {
  name     = "${local.project_id}-terraform-state"
  location = local.gcs_location

  lifecycle {
    prevent_destroy = true
  }
}
# raw data source files
resource "google_storage_bucket" "raw" {
  name     = "${local.project_id}-raw"
  location = local.gcs_location
}
# processed datasets for training ML models
resource "google_storage_bucket" "datasets" {
  name     = "${local.project_id}-datasets"
  location = local.gcs_location
}

# GKE K8s Cluster
resource "google_container_cluster" "main" {
  name             = "main"
  enable_autopilot = true

  private_cluster_config {
    # disable public internet access to worker nodes
    enable_private_nodes    = true
    enable_private_endpoint = false # allow public access to k8s endpoint
  }
}
