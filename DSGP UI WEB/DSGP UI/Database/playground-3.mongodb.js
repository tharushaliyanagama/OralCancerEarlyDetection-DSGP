use("User_Details");
db.Username.insertMany([
    {
        "Username": "John",
        "Password": "John123",
        "Email": "john@gmail.com"}
]);

console.log(db.Username.find().pretty());