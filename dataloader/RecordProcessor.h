#pragma once
ref class RecordProcessor
{
private:
	Npgsql::NpgsqlConnection^ connection;
	int ProcessRaRecord(System::String^ record);
	int ProcessSeRecord(System::String^ record);
	int ProcessUmRecord(System::String^ record);

public:
	static const int PROCESS_ERROR = -1;
	static const int PROCESS_SUCCESS = 0;
	static const int PROCESS_SKIP = 1;

	bool Initialize();
	RecordProcessor(Npgsql::NpgsqlConnection^ connection);
	int ProcessRecord(System::String^ record);
};

