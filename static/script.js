$(document).ready(function(){
    $("#reorder").click(function(event){
        var input = $("#user-input").val();
        $.ajax({
              type: "POST",
              url: '/learning',
              data: JSON.stringify({userInput: input}),
              contentType: 'application/json',
              success: function(response){
                    console.log(response.results);
                   $("#results").text(response.results);
                },
          });
    });
});

// $(document).ready(function(){
//         $("#submit").click(function(event){
//                 var uInput = $("#user-input").val();
//                 $.ajax({
//                  type: "POST", //request type,
//                  url:'/learning', //the page containing python script
//                  data: JSON.stringify({userInput: uInput}),
//                 contentType: 'application/json',
//                         success: function(response){
//                         console.log(response.average);
//                             $("#results").text(response.average);
//                           },
//                     });
//                 });
//             });