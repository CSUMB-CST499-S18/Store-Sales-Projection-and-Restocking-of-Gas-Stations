<?php

function getDatabaseConnection(){
    $host = "us-cdbr-iron-east-05.cleardb.net";
    $dbname = "heroku_061d37f76a72480";
    $username = "ba2b0c564eb8c7";
    $password = "c29e3084";
    
    //Creates a database connection
    $dbConn = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    
    // Setting Errorhandling to Exception
    $dbConn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION); 
    
    return $dbConn;    
}

?>