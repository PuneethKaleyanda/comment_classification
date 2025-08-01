$(document).ready(function() {
    $('#comment-form').on('submit', function(e) {
        e.preventDefault();
        
        const comment = $('#comment').val().trim();
        const resultDiv = $('#result');
        const resultList = $('#result-list');
        
        // Clear previous results
        resultList.empty();

        if (comment === '') {
            alert("Please enter a comment.");
            return;
        }

        // Send POST request to Flask backend
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: { comment: comment },
            success: function(response) {
                // Show the result section
                resultDiv.show();
                
                // Display the result
                $.each(response, function(key, value) {
                    const listItem = $('<li class="list-group-item d-flex justify-content-between align-items-center"></li>');
                    listItem.text(`${key.charAt(0).toUpperCase() + key.slice(1)}:`);
                    const badge = $('<span class="badge badge-pill"></span>');
                    badge.text(value ? 'Positive' : 'Negative');
                    badge.addClass(value ? 'badge-success' : 'badge-danger');
                    listItem.append(badge);
                    resultList.append(listItem);
                });
            },
            error: function(error) {
                console.error('Error:', error);
            }
        });
    });
});
