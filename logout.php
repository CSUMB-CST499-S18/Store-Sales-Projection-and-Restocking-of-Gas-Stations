<?php
session_start();
session_destroy();

header("Location: login.php");


?>

<!DOCTYPE html>
<html>
    <head>
         <a href='index.php'><button type="button" class="btn btn-default btn-lg">
            <span class="glyphicon glyphicon-pencil" aria-hidden="true"></span> Return to Home Page

    </head>
    <body>

    </body>
</html>
