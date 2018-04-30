$(document).ready(function(){
    $("#reorder").click(function(event){
        console.log("Entered Ajax Function")
        var input = $("#user-input").val();
        $.ajax({
              type: "POST",
              dataType: 'text',
              url: '/learning',
              data: JSON.stringify({userInput: input}),
              contentType: 'application/json',
              success: function(response){
                    console.log("Made it");
                    console.log(response.formatted_result);
                   $("#results").text(response.formatted_result);
                },
                 error: function(request,status, message) {
                        console.log(request);
                        console.log("----");
                        console.log(status);
                        }
          });
    });
});
