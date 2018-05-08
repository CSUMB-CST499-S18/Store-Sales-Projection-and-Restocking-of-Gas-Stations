<script>
     function validatePOS() {
            $.ajax({
                type: "GET",
                url: "https://capstone-frontend-kylekern.c9users.io/checkPOS.html",
                dataType: "json",
                data: {
                    'posNum': $('#posNum').val(),
                    'action': 'validate-username'
                },
                success: function(data,status) {
                    debugger;
                    if (data.length>0) {
                        $('#username-valid').html("POS code not in system");
                        $('#username-valid').css("color", "red");
                    } else {
                        $('#username-valid').html("POS code found!"); 
                        $('#username-valid').css("color", "green");
                    }
                  },
                complete: function(data,status) { 
                    //optional, used for debugging purposes
                    //alert(status);
                }
            });
        }
    function passvalue(description) {
    
    
  
   
        var item = description;
        console.log(item);
        console.log("Entered Ajax Function")
        $('#loadingmessage').show();
        
        // var input = $("#user-input").val();
        $.ajax({
              type: "POST",
              dataType: 'text',
              url: 'http://capstone2-jmts23.c9users.io:8081/learning',
              data: JSON.stringify({userInput: item}),
              contentType: 'application/json',
              success: function(response){
                    console.log("Made it");
                    $('#loadingmessage').hide();
                    output = response;
                    newoutput = JSON.parse(output);
                    console.log(newoutput);
                    // var newOutput = output.substring(1, output .length-2);
                    // console.log(newOutput);
                    console.log("---");
                    console.log(newoutput.results[0].Date);
                    console.log(newoutput.results[0].Prediction);
                    console.log(newoutput.results.length);

                    
                    $('#calendar').fullCalendar({
                    defaultView: 'month',
                    events: newoutput.results
                    // [
                    //     {
                    //       title  : newoutput.results[x].Prediction,
                    //       start  : newoutput.results[x].Date,
                    //       allDay : true
                    //     },
                    //     {
                    //       title  : 'event2',
                    //       start  : '2010-01-05',
                    //       end    : '2010-01-07'
                    //     },
                    //     {
                    //       title  : 'event3',
                    //       start  : '2010-01-09T12:30:00',
                    //       allDay : true // will make the time show
                    //     }
                    // ]
                    ,
                    height: 300
                });
              }
                
            
                   
                ,
                 error: function(request,status, message) {
                        console.log(request);
                        console.log("----");
                        console.log(status);
                        }
          });
 

    
}

</script>

<?php

session_start();
if(!isset($_SESSION['username'])){
   header("Location:index.html");
}



include 'dbConnection.php';

$con = getDatabaseConnection('heroku_87e7042268995be');


function listUsers() {
    global $con;
    $namedParameters = array();
    $results = null;
    $sql = "SELECT *
            FROM topsales";
    $stmt = $con -> prepare ($sql);
    $stmt -> execute($namedParameters);
    $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
    echo "<table id=\"table1\">
        <tr>
 	    <th> Description &nbsp &nbsp  &nbsp &nbsp  &nbsp&nbsp &nbsp  &nbsp &nbsp  &nbsp</th>
 	    <th> PosCode &nbsp &nbsp  &nbsp &nbsp  &nbsp &nbsp  &nbsp &nbsp  &nbsp&nbsp</th>
 	    <th> Total Sold &nbsp &nbsp  &nbsp &nbsp  &nbsp &nbsp  &nbsp &nbsp </th>
 	    <th> Total Stock &nbsp &nbsp  &nbsp&nbsp &nbsp  &nbsp &nbsp  &nbsp &nbsp  &nbsp</th>
        </tr>";
    foreach($results as $result) {
         echo "<tr>";
        echo "<td><a href=# onclick=\"passvalue('".$result['description']."')\">".$result['description']."</a></td>".
        "<td>".$result['PosCode']."</td>".
        "<td>".$result['salesQuntity']."</td>".
        "<td>".$result['salesAmount']."</td>";
        echo "</tr>";
    }
    echo "</table>";
}
    //SETTING UP ARRAy TO BE PASSED INTO PREDICTION FUNCTION
    global $con;
    $namedParameters = array();
    $results = null;
    $sql = "SELECT Description
            FROM sales";
    $stmt = $con -> prepare ($sql);
    $stmt -> execute($namedParameters);
    $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
    // availableItems is the array name used for the prediction
    $availableItems = array();
    foreach($results as $result){
        array_push($availableItems,$result['Description']);
    }
    sort($availableItems);
    //END
?>

<!DOCTYPE html>
<html>
    <head>
        <title>S.S.P.A.R</title>
        <meta charset="utf-8">
        <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <!-- jQuery library -->
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <!-- Latest compiled JavaScript -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        <link rel="stylesheet" href="./css/styles.css" type="text/css" />
        <!--Prediction dependencies-->
        <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
        <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
        <!--Converting from php array to usabel javascript array-->
        <script>
              jArray= <?php echo json_encode($availableItems); ?>;
        </script>
        <!--Full Calendar Dependencies-->
        <link rel='stylesheet' href='fullcalendar-3.9.0/fullcalendar.css' />
        <script src='fullcalendar-3.9.0/lib/moment.min.js'></script>
        <script src='fullcalendar-3.9.0/fullcalendar.js'></script>
    </head>
    
    <body>
        <div id='loadingmessage' style='display:none'>
      <img src='loadinggraphic.gif'/>
</div>
    <section class="container">
      <div class="sscs"> 
          <img src="./img/sscs-logo.png" alt="SSCS">
      </div> 
      <div class="search">
      <form action="users.php" method="GET">
        <input id="search" type="text" placeholder="Type here">
        <!--script to predict text on user input-->
        <script>$( "#search" ).autocomplete({source: jArray});</script>
        <input id="submit" type="submit" value="Search">
      </form>
      <form action="users.php" method="GET">
        <input onchange="validatePOS();"input id="search" type="text" placeholder=" POS search">
        <input id="submit" type="submit" value="Search">
      </form>
      <form action="about.html">
        <input type="submit" value="About Us">
      </form>
      <form action="logout.php">
        <input type="submit" value="Logout" />
      </form>
     <!--Calendar is rendered here. Events need to be changed to reflect the actual contents-->
     </div>
      <div id = "calendar">
         <script>
         </script>
     </div>
   </section>
      <div class="clear"></div>
     
<section class="container2">
    <center>
   <h2 class="sub-header">Top Selling Items</h2>
   <div id="results">
       
   </div>
   <div id=table>
         <?php 
    
 	  listUsers();
    ?>
    </div>

</center>
</section>
    </body>
</html>
