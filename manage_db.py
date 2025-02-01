import logging
from pymilvus import connections, utility, Collection
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MilvusManager:
    def __init__(self):
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

    def list_collections(self):
        """List all collections in Milvus"""
        collections = utility.list_collections()
        logger.info(f"Found {len(collections)} collections:")
        for coll in collections:
            logger.info(f"- {coll}")
        return collections

    def delete_collection(self, collection_name: str):
        """Delete a specific collection"""
        try:
            if collection_name in utility.list_collections():
                # Drop collection
                Collection(collection_name).drop()
                logger.info(f"Successfully deleted collection: {collection_name}")
                
                # Only remove tracking file if we're deleting markdown collection
                if collection_name == "markdown_collection" and os.path.exists('processed_files.json'):
                    os.remove('processed_files.json')
                    logger.info("Removed processed files tracking")
                
                return True
            else:
                logger.info(f"Collection {collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Manage Milvus Vector Database')
    parser.add_argument('--list', action='store_true', help='List all collections')
    parser.add_argument('--delete', type=str, help='Delete specified collection by name')
    args = parser.parse_args()

    manager = MilvusManager()

    if args.list:
        manager.list_collections()
    elif args.delete:
        confirm = input(f"Are you sure you want to delete collection '{args.delete}'? This cannot be undone! (yes/no): ")
        if confirm.lower() == 'yes':
            manager.delete_collection(args.delete)
        else:
            print("Operation cancelled")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
