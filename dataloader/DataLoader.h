﻿#pragma once
ref class Config;
ref class RecordProcessor;
ref class DataLoader {
private:
    ref struct JVOpenParams {
        System::String^ dataSpec;
        System::String^ fromDate;
        int option;
        JVOpenParams(System::String^ dataSpec, System::String^ fromDate, int option) {
            this->dataSpec = dataSpec;
            this->fromDate = fromDate;
            this->option = option;
        }
    };
    AxJVDTLabLib::AxJVLink^ jvlink;
    Config^ config;
    Npgsql::NpgsqlConnection^ connection;
    RecordProcessor^ recordProcessor;
    bool Initialize();
    bool ProcessChunk(JVOpenParams^);
    bool ProcessRealtimeChunk(System::String^ dataSpec, System::String^ date);
    int WaitForDownloadCompletion(int downloadCount);
    bool InitializeDatabase(bool deleteIfExists);
    bool CreateTimestampTable();
    bool UpdateLastFileTimestamp(System::String^ timestamp);
    bool RollbackTransaction(Npgsql::NpgsqlTransaction^ transaction);
    array<System::String^>^ args;
    bool ExecuteSetup();
    bool ExecuteUpdate();
    bool ExecuteRealtime();
public:
    DataLoader(AxJVDTLabLib::AxJVLink^, array<System::String^>^);
    bool Execute();
};

