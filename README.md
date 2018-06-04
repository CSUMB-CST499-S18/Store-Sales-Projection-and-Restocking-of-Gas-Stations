
THE MOST UP TO DATE BRANCH IS THE final_backend BRANCH.IT HAS NOT BEEN MERGED TO MASTER YET.

## Store Sales Projection and Restock (SSPaR)

Steps to get application running:

1) Open a terminal, cd into the project and type 'python flask_server.py'
- This is being ran on a different port(8081) than the default 8080 that c9 runs on.

2) Before running admin.php, the ajax call in admin.php url link should be changed to the url of the c9 link that the server is being hosted on (i.e http://capstone-(insert username).c9users.io:8081/learning) on both ajax functions within admin.php.

3)open up admin.php with credentials
username: admin
password: password

4) When the admin.php is up, remove the s from https in your URL link in the browser.

5) You will get a CORS error if you try to make requests to the Flask server, to work around that you can install a google chrome plugin Allow-Control-Allow-Origin: *

From there you should be good to go.
