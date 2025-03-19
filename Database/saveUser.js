const mongoose = require('mongoose');

// Connect to the MongoDB database
mongoose.connect('mongodb://localhost:27017/User_details', {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('Connection error:', err));

// Define the schema for storing usernames
const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true }
});

// Create a model for the collection
const User = mongoose.model('username', userSchema);  // Collection name: 'username'

// Function to save a new username
async function saveUsername(userInput) {
    try {
        const newUser = new User({ username: userInput });
        await newUser.save();
        console.log('Username saved successfully:', userInput);
    } catch (error) {
        console.error('Error saving username:', error.message);
    } finally {
        mongoose.connection.close(); // Close connection after saving
    }
}

// Example: Save a username (replace with user input)
const enteredUsername = 'john_doe';  // Replace with actual user input
saveUsername(enteredUsername);
