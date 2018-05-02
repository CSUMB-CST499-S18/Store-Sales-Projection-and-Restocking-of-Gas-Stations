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
        echo "<td>". $result['description'] . "</a></td>".
        "<td><a href=\"info.php?name=".$result['POScode']."\">".$result['PosCode']."</td>".
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
    </head>
    
    <body>
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
     </div>
   </section>
      <div class="clear"></div>
     
<section class="container2">
    <center>
   <h2 class="sub-header">Top Selling Items</h2>
   <div id=table>
         <?php 
 	  listUsers();
    ?>
    </div>
</center>
</section>
    </body>
</html>
