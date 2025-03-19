const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = 3000;
app.use(cors());

app.use(bodyParser.json());

// Middleware to parse form data
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files (CSS, images, etc.)
app.use(express.static(path.join(__dirname, "static")));
app.use(express.static(path.join(__dirname, "templates")));

// Connect to MongoDB
mongoose
  .connect("mongodb+srv://sithmi:ef76hPUdVohpVAZX@cluster0.s4n7e.mongodb.net/User_Details", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("Error connecting to MongoDB:", err));

// Define the schema and model for the User collection
const userSchema = new mongoose.Schema({
  username: { type: String, required: true },
  password: { type: String, required: true },
  email: { type: String, required: true, unique: true },
});

const User = mongoose.model("Username", userSchema);


// Serve the signup page (if needed)
 app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'signup.html'));
 });

// Route to handle signup form submission
app.post("/signup", async (req, res) => {
  const { username, password, email } = req.body;

  try {
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists. Please login." });
    }

    // Save new user
    const newUser = new User({ username, password, email });
    await newUser.save();
    
    // Respond with success
    res.status(201).json({ message: "User signed up successfully! Go to Login." });
  } catch (error) {
    console.error("Error saving user:", error.message);
    res.status(500).json({ message: "Error saving user." });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
