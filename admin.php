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
    $sql = "SELECT Description
            FROM sales";
    $stmt = $con -> prepare ($sql);
    $stmt -> execute($namedParameters);
    $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
    echo "<table id=\"table1\">
        <tr>
 	    <th> description </th>
        </tr>";
    foreach($results as $result) {
        echo "<tr>";
        echo "<td><a href=\"info.php?name=". "&Description=" . 
            $result['Description'] .
            "\">" . $result['Description'] . "</a></td>";
        echo "</tr>";
    }
    echo "</table>";
}
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
    </head>
    
    <body>
    <section class="container">
      <div class="sscs"> 
          <img src="./img/sscs-logo.png" alt="SSCS">
      </div> 
      <div class="search">
      <form action="users.php" method="GET">
        <input id="search" type="text" placeholder="Type here">
        <input id="submit" type="submit" value="Search">
      </form>
      <h5>
          POSsearch
      </h5>
      <form action="users.php" method="GET">
        <input id="search" type="text" placeholder="Type here">
        <input id="submit" type="submit" value="Search">
      </form>
     <form action="about.html">
        <input type="submit" value="About Us">
      </form>
       <br/>
     <form action="logout.php">
     <!-- <form action="404.html">
        <input type="submit" value="reorder soon">
      </form>
      -->
      <form action="logout.php">
        <input type="submit" value="Logout" />
      </form>
     </div>
   </section>

      <div class="clear"></div>
     
<section class="container2">
    <center>
   <h2 class="sub-header">Items</h2>
   
         <?php 
 	  listUsers();
    ?>
</center>
</section>
    </body>
</html>
