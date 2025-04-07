const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const path = require("path");
const cors = require("cors");
const multer = require("multer");

const app = express();
const PORT = 3001;
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files
app.use(express.static(path.join(__dirname, "static/CSS")));
app.use(express.static(path.join(__dirname, "templates")));
app.use(express.static(path.join(__dirname, "static/images")));
app.use(express.static("uploads"));

// Connect to MongoDB
mongoose.connect("mongodb+srv://sithmi:ef76hPUdVohpVAZX@cluster0.s4n7e.mongodb.net/User_Details", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("Error connecting to MongoDB:", err));

// Define User schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true },
  password: { type: String, required: true },
  email: { type: String, required: true, unique: true },
});

const User = mongoose.model("User", userSchema);

// Serve Pages
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'templates', 'home.html')));
app.get('/login', (req, res) => res.sendFile(path.join(__dirname, 'templates', 'login.html')));
app.get('/predict', (req, res) => res.sendFile(path.join(__dirname, 'templates', 'dataupload.html')));

// Signup API
app.post("/signup", async (req, res) => {
  const { username, password, email } = req.body;

  try {
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists. Please login." });
    }

    const newUser = new User({ username, password, email });
    await newUser.save();

    res.status(200).json({ message: "Sign up successful!" });
  } catch (error) {
    console.error("Signup error:", error.message);
    res.status(500).json({ message: "Error saving user." });
  }
});

// Login API
app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });

    if (!user || user.password !== password) {
      return res.status(401).json({ message: "Invalid email or password." });
    }

    res.redirect('/predict');
  } catch (error) {
    console.error("Login error:", error.message);
    res.status(500).json({ message: "An error occurred during login." });
  }
});

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));