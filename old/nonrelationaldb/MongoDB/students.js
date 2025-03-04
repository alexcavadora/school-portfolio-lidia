import { config } from "dotenv";
import { Collection, MongoClient } from "mongodb";
config();
console.log(process.env.DB_URI);

const getStudents = async (collection) => {
  return await collection.find().toArray();
};
const createStudent = async (collection) => {
  const student = {
    name: "Alejandro Alonso",
    birthdate: new Date(2002, 5, 26),
    adress: {
      street: "revolución",
      number: 337,
      city: "Salamanca",
      state: "GTO",
    },
  };
  await collection.insertOne(student);
};
export async function studentsCrudOperations() {
  const uri = process.env.DB_URI;
  let mongoClient;
  try {
    mongoClient = await connectToMongoDB(uri);
    const db = mongoClient.db("university");
    const collection = db.collection("students");
    await createStudent(collection);
    console.log("List of students");
    console.log(await getStudents(collection));
  } finally {
    mongoClient?.close();
  }
}

async function connectToMongoDB(uri) {
  let mongoClient;
  try {
    mongoClient = new MongoClient(uri);
    console.log("Connecting to MongoDB...");
    await mongoClient.connect();
    console.log("Connected to MongoDB");
    return mongoClient;
  } catch (error) {
    console.log("Connection failed:", error);
  }
}
//db = connect("mongodb://root:root@localhost:27017/admin");
//db.getSiblingDB("mydb").mycollection.insert({ name: "John Doe", age: 30 });
