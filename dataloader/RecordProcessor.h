#pragma once
ref class RecordProcessor
{
private:
	Npgsql::NpgsqlConnection^ connection;
	int ProcessRARecord(array<System::Byte>^ record);
	int ProcessSERecord(array<System::Byte>^ record);
	int ProcessUMRecord(array<System::Byte>^ record);
	int ProcessHNRecord(array<System::Byte>^ record);
	int ProcessCHRecord(array<System::Byte>^ record);
	int ProcessKSRecord(array<System::Byte>^ record);
	int ProcessCSRecord(array<System::Byte>^ record);
	int ProcessWERecord(array<System::Byte>^ record);
	int ProcessAVRecord(array<System::Byte>^ record);
	int ProcessJCRecord(array<System::Byte>^ record);
	int ProcessTCRecord(array<System::Byte>^ record);
	int ProcessCCRecord(array<System::Byte>^ record);
	int ProcessWHRecord(array<System::Byte>^ record);
	System::String^ ByteSubstring(array<System::Byte>^ bytes, int byteStartIndex, int byteLength);

public:
	static const int PROCESS_ERROR = -1;
	static const int PROCESS_SUCCESS = 0;
	static const int PROCESS_SKIP = 1;

	bool Initialize();
	RecordProcessor(Npgsql::NpgsqlConnection^ connection);
	int ProcessRecord(array<System::Byte>^ record);
};

