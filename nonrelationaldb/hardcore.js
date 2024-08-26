const http = require("node:http");
const desiredPort = process.env.PORT ?? 8080;
// si ya tenemos un puerto en trabajo, lo usamos, de lo contrario, utilizaremos el 8080
const processRequest = (request, response) => {
  response.setHeader("Content-type", "text/html; charset=utf-8");
  if (request.url === "/") {
    response.end("<h1> Bienvenidos !</h1>");
  } else if (request.url === "/hola") {
    response.end("<h2> hola! </h2>");
  } else {
    response.end("<h1>404</h1>");
  }
};
const server = http.createServer(processRequest);

server.listen(desiredPort, () => {
  console.log(`Server listening on Port: http://localhost:${desiredPort}`);
});
