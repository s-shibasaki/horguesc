ref class Config {
private:
	// DataLoader
	System::String^ _sid;
	int _startYear;
	bool _deleteDatabase;

	// Database
	System::String^ _dbHost;
	int _dbPort;
	System::String^ _dbName;
	System::String^ _dbUsername;
	System::String^ _dbPassword;

public:
	// Dataloader
	property System::String^ Sid {System::String^ get() { return _sid; }};
	property int StartYear {int get() { return _startYear; }};
	property bool DeleteDatabase {bool get() { return _deleteDatabase; }};

	// Database
	property System::String^ DbHost {System::String^ get() { return _dbHost; }};
	property int DbPort {int get() { return _dbPort; }};
	property System::String^ DbName {System::String^ get() { return _dbName; }};
	property System::String^ DbUsername {System::String^ get() { return _dbUsername; }};
	property System::String^ DbPassword {System::String^ get() { return _dbPassword; }};

	Config();
	bool Load(System::String^ iniFile);
};

