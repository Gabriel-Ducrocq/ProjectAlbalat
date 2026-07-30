#!/bin/bash
apptainer exec --bind "$(pwd)/albalat/data/processed/qdrant_storage:/qdrant/storage" albalat/data/raw/qdrant_v1.18.3.sif /qdrant/qdrant &

QDRANT_SIF="albalat/data/raw/qdrant_v1.18.3.sif"
QDRANT_STORAGE="$(pwd)/albalat/data/processed/qdrant_storage"

# Start Qdrant in the background
apptainer exec --bind "${QDRANT_STORAGE}:/qdrant/storage" "${QDRANT_SIF}" /qdrant/qdrant &
QDRANT_PID=$!

# Ensure it's killed no matter how this script exits
cleanup() {
	  echo "Stopping Qdrant (PID ${QDRANT_PID})..."
	    kill "${QDRANT_PID}" 2>/dev/null || true
	      wait "${QDRANT_PID}" 2>/dev/null || true
      }
      trap cleanup EXIT

      # Wait until Qdrant's REST API responds before continuing
      echo "Waiting for Qdrant to become ready..."
      until curl -sf http://localhost:6333/readyz > /dev/null 2>&1; do
	        sleep 1
	done
	echo "Qdrant is up."


source .venv/bin/activate
python albalat/scripts/embeddings_to_vector_db.py albalat/data/raw/qdrant.yaml
