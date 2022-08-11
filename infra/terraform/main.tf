#
# sortin-hat
# Terraform Infrastructure
#

locals {
  project_id = "sss-sortin-hat"
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

# GCS bucket to as terraform state backend
resource "google_storage_bucket" "tf_state" {
  name     = "${local.project_id}-terraform-state"
  location = "ASIA-SOUTHEAST1" # Singapore

  lifecycle {
    prevent_destroy = true
  }
}
