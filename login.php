<!DOCTYPE html>
<html>
    <head>
         <link rel="stylesheet" href="css/styles.css" type="text/css" />
        <title>Admin Login Page</title>
    </head>
    <body>

       <h1> - Admin Login -</h1>
        <form action="processLogin.php"  method="post">
            Username: <input type="text" name="username" /> <br />
            Password: <input type="password" name="password"  />
            <input type="submit" name="loginForm" />
        </form>
        <form action = "processLogin.php" method ="post">
        <input type="submit" value="Return to Homepage" /> 
        </form>

    </body>
</html>
