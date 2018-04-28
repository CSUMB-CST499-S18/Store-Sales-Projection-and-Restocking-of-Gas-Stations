<!DOCTYPE html>
<html>
    <head>
       <meta charset="utf-8">
        <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <!-- jQuery library -->
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <!-- Latest compiled JavaScript -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        <link rel="stylesheet" href="./css/styles.css" type="text/css" />
        <title>Admin Login Page</title>
    </head>
    <body>
    <section class="admin">
       <h1> - Admin Login -</h1>
        <form action="processLogin.php"  method="post">
            Username: <input type="text" name="username" /> <br />
            Password: <input type="password" name="password"  />
            <br>
            <input type="submit" name="loginForm" />
        </form>
    </section>
    </body>
</html>
