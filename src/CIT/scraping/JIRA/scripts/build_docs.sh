MODEL_NAME="llama3.3"
OUTPUT_DIR="./src/CIT/scraping/JIRA/jira_documents_llama70b"
WAITING_TIME=11000 # seconds


#countdown to start the training script
for ((i=WAITING_TIME; i>0; i--)); do
  echo -ne "Starting in $i seconds...\r"
  sleep 1
done


python ../build_documents_from_tickets.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME"

echo "Document building completed."