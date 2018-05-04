<?php
session_start();  

include 'dbConnection.php';

$dbConn = getDatabaseConnection('heroku_061d37f76a72480');

$username = $_POST['username'];
$password = $_POST['password']; 

$sql = "SELECT * 
        FROM admin
        WHERE userName = :username
          AND password = :password";

$namedParameters = array();          
$namedParameters[':username'] = $username;  
$namedParameters[':password'] = $password;  

$statement = $dbConn->prepare($sql);
$statement->execute($namedParameters);
$record = $statement->fetch(PDO::FETCH_ASSOC);
print_r($record);

    if (empty($record)) { 
        header('Location: index.html');
        echo "Wrong username or password!";
        echo "<a href='index.html'> Try again </a>";
        
    } else {
        session_start();
        $_SESSION['username'] = $record['username'];
        $_SESSION['adminName'] = $record['firstName'] . " " . $record['lastName'];
        
        header('Location: admin.php');  //redirects to another program        
        
    }
?>
