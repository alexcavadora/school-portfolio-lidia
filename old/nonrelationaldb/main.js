const express = require("express");
const desiredPort = process.env.PORT ?? 8080;
// si ya tenemos un puerto en trabajo, lo usamos, de lo contrario, utilizaremos el 8080
const app = express();
app.use(express.json());
const getPokemonData = async (pokemonName) => {
  const url = `https://pokeapi.co/api/v2/pokemon/${pokemonName}`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Http error!: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    throw error;
  }
};
app.get("/pokemon/charmander", async (request, response) => {
  const pokemonData = await getPokemonData("charmander");
  response.json(pokemonData);
});
app.get("/", (request, response) => {
  response.json((msg = "hola!"));
});
// app.use((req,res {
//   response
// }))
app.listen(desiredPort, () => {
  console.log(`Listening on http://localhost:${desiredPort}`);
});
