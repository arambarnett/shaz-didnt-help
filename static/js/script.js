$(document).ready(function () {
    let selectedStyle = '';
    let downloadUrl = '';
  
    $("#uploadBtn").click(function () {
      if ($(this).find('.state-1-text').hasClass('active')) {
        $("#fileInput").click();
      } else if ($(this).find('.state-4-text').hasClass('active') && downloadUrl) {
        window.location.href = downloadUrl;
      }
    });
  
    $("#fileInput").change(function (e) {
      const files = e.target.files;
      if (files.length > 0) {
        // Show selected files count
        let fileNames = Array.from(files).map(file => file.name).join(', ');
        $('#fileList').html(`Selected files: ${fileNames}`);
        
        $(".primary-btn").removeClass("state-1").addClass("state-2");
        $(".state-1-text").removeClass("active");
        $(".state-2-text").addClass("active");
        $(".file-details").show();
        $("#styleOptions").show();
      }
    });
  
    $(".feature").click(function () {
      selectedStyle = $(this).data('style');
      $(".feature").removeClass("selected");
      $(this).addClass("selected");
      $(".selected-style").text(`Selected style: ${selectedStyle}`);
      $("#styleOptions").hide();
      uploadFiles();
    });
  
    async function uploadFiles() {
        const files = document.getElementById('fileInput').files;
        
        try {
            const fileUrls = [];
            const totalFiles = files.length;
            
            // Show processing state
            $('.primary-btn').removeClass('state-2').addClass('state-3');
            $('.state-2-text').removeClass('active');
            $('.state-3-text').addClass('active');
            $('.progress-container').show();
            
            console.log(`Starting upload of ${totalFiles} files`);
            
            for (let i = 0; i < totalFiles; i++) {
                const file = files[i];
                console.log(`Starting upload for file ${i + 1}/${totalFiles}: ${file.name}`);
                $('#current-stage').text(`Uploading file ${i + 1} of ${totalFiles}...`);
                
                // Get signed URL
                const response = await fetch('/api/upload-url', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        fileName: file.name,
                        contentType: file.type || 'video/quicktime'
                    })
                });

                if (!response.ok) {
                    console.error('Failed to get upload URL:', response.status);
                    throw new Error(`Failed to get upload URL: ${response.status}`);
                }

                const data = await response.json();
                console.log('Upload URL response:', data);

                if (!data || !data.url) {
                    console.error('Invalid response data:', data);
                    throw new Error('No upload URL received from server');
                }

                // Upload to GCS
                console.log('File details:', {
                    name: file.name,
                    type: file.type,
                    size: file.size
                });
                console.log('Using signed URL:', data.url);

                const uploadResponse = await fetch(data.url, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': file.type || 'video/quicktime',
                        'Content-Length': file.size.toString(),
                        'x-goog-content-length-range': '0,5368709120'
                    },
                    body: file
                });

                console.log('Upload response:', uploadResponse);

                if (!uploadResponse.ok) {
                    const errorText = await uploadResponse.text();
                    console.error('Upload failed:', errorText);
                    throw new Error(`Upload failed: ${uploadResponse.status} - ${errorText}`);
                }

                fileUrls.push(data.fileName);
                console.log(`Successfully uploaded file ${i + 1}/${totalFiles}`);
            }

            // Process the uploaded files
            $('#current-stage').text('Processing videos...');
            const processResponse = await fetch('/api/process_from_storage', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fileUrls: fileUrls,
                    style: selectedStyle || 'Fast & Energized'
                })
            });

            if (!processResponse.ok) {
                throw new Error(`Processing failed: ${processResponse.status}`);
            }

            const result = await processResponse.json();
            if (result.status === 'success') {
                downloadUrl = result.download_url;
                $('.primary-btn').removeClass('state-3').addClass('state-4');
                $('.state-3-text').removeClass('active');
                $('.state-4-text').addClass('active');
                $('#startOverBtn').show();
                $('#current-stage').text('Processing complete!');
            } else {
                throw new Error(result.error || 'Processing failed');
            }
            
        } catch (error) {
            console.error('Full error details:', error);
            $('.state-3-text').text('Error - Try Again');
            $('#current-stage').text(`Error: ${error.message}`);
            alert(`Error: ${error.message}`);
        }
    }
  
    $('#startOverBtn').click(function() {
      // Reset UI
      $('.primary-btn').removeClass('state-2 state-3 state-4').addClass('state-1');
      $('.state-1-text').addClass('active');
      $('.state-2-text, .state-3-text, .state-4-text').removeClass('active');
      $('#styleOptions').hide();
      $('#startOverBtn').hide();
      $('.file-details').hide();
      $('.progress-container').hide();
      $('#fileInput').val('');
      selectedStyle = '';
      downloadUrl = '';
    });
});
  