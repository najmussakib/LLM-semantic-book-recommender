from tqdm.auto import tqdm
import time

from langchain_chroma import Chroma

# Function to create embeddings in batches: to avoid rate limiting error
def create_embeddings_in_batches(documents, embedding_model, batch_size=100, wait_time=1):

    # Split docs into batches
    batches = [ documents[i:i + batch_size] for i in range(0, len(documents), batch_size) ]

    # Create a new Chroma Collection
    db = None

    # Process each batch
    for i, batch in enumerate(tqdm(batches)):
        try:
            # For Ist batch, create the DB
            if i == 0:
                db = Chroma.from_documents(batch, embedding=embedding_model)
            # For subsequent batches, add to the existing DB
            else:
                db.add_documents(batch)

            # Wait between batches to avoid rate limits
            if i < len(batches) - 1:    # Don't need to wait after the last batch
                print(f"Waiting for {wait_time} seconds before next batch...")
                time.sleep(wait_time)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")

            # If there is an error, wait longer and try again
            print(f"Waiting for {wait_time*2} seconds before retrying...")
            time.sleep(wait_time*2)
            try:
                if db is None:
                    db = Chroma.from_documents(batch, embedding=embedding_model)
                else:
                    db.add_documents(batch)
            except Exception as e2:
                print(f"Error processing batch {i}: {e2}")
                break

    return db