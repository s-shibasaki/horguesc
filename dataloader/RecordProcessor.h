#pragma once
ref class RecordProcessor
{
private:
public:
	static const int PROCESS_ERROR = -1;
	static const int PROCESS_SUCCESS = 0;
	static const int PROCESS_SKIP = 1;

	RecordProcessor();
	~RecordProcessor();

	int ProcessRecord(System::String^ record);
	bool InitializeDatabase();
};

