const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const path = require("path");
const cors = require("cors");
const multer = require("multer");

const app = express();
const PORT = 3000;
app.use(cors());
app.use(bodyParser.json());
app.use(express.static("uploads")); // Serve uploaded files

// Middleware to parse form data
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files (CSS, images, etc.)
app.use(express.static(path.join(__dirname, "static/CSS")));
app.use(express.static(path.join(__dirname, "templates")));
app.use(express.static(path.join(__dirname, "static/images")));

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

const User = mongoose.model("User", userSchema);

// Serve signup page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'signup.html'));
});

// Serve login page
app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

// Serve dataupload page
app.get('/predict', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'dataupload.html'));
});

// Handle login
app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    // Check if user exists
    const user = await User.findOne({ email });

    if (!user || user.password !== password) {
      return res.status(401).json({ message: "Invalid email or password." });
    }

    // Successful login - Redirect to dataupload.html
    res.redirect('/predict');
  } catch (error) {
    console.error("Login error:", error.message);
    res.status(500).json({ message: "An error occurred during login." });
  }
});

// Handle signup
app.post("/signup", async (req, res) => {
  const { username, password, email } = req.body;

  try {
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      console.error("Signup error: User already exists with this email.");
      return res.status(400).json({ message: "User already exists. Please login." });
    }

    
    // Save new user
    const newUser = new User({ username, password, email });
    await newUser.save();

    console.log("User signed up successfully:", newUser);

    // Successful signup - Redirect to /predict
    return res.redirect("/predict");
  } catch (error) {
    console.error("Error saving user:", error.message);
    return res.status(500).json({ message: "Error saving user." });
  }
});


// Temporary in-memory storage for patient data
let patientData = {};

// Handle patient data submission
app.post("/submit-patient-data", (req, res) => {
    patientData = req.body;
    console.log("Patient Data Received:", patientData);
    res.json({ message: "Patient data saved successfully!" });
});

// Multer setup for image upload
const storage = multer.diskStorage({
    destination: "./uploads/",
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname)); // Unique filename
    }
});

const upload = multer({ storage });

// Handle image upload
app.post("/upload-image", upload.single("image"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: "No file uploaded" });
    }
    res.json({ imageUrl: `/uploads/${req.file.filename}` });
});

// Get saved patient data
app.get("/get-patient-data", (req, res) => {
    res.json(patientData);
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
