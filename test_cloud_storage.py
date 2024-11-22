from google.cloud import storage
import os
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cloud_storage():
    try:
        # Initialize the client
        storage_client = storage.Client()
        
        # Test bucket access
        bucket_name = 'shaz_didnt_help_bucket_1'
        bucket = storage_client.bucket(bucket_name)
        
        # Test file upload using shazwithhotdog.png
        test_file_path = 'static/images/shazwithhotdog.png'
        if not os.path.exists(test_file_path):
            logger.error(f"Test file not found: {test_file_path}")
            return False
            
        blob = bucket.blob('test/shazwithhotdog.png')
        blob.upload_from_filename(test_file_path)
        logger.info(f"Successfully uploaded {test_file_path} to {bucket_name}")
        
        # Test file download
        download_path = 'test_download.png'
        blob.download_to_filename(download_path)
        logger.info(f"Successfully downloaded test file from {bucket_name}")
        
        # Test generating signed URL
        url = blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=15),
            method='GET'
        )
        logger.info(f"Successfully generated signed URL: {url}")
        
        # Clean up downloaded file
        os.remove(download_path)
        logger.info("Successfully cleaned up test files")
        
        # Don't delete the original from bucket for verification
        logger.info("Test file remains in bucket for verification")
        
        return True
        
    except Exception as e:
        logger.error(f"Cloud Storage test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_cloud_storage() 