#pragma once
ref class Config;
ref class DataLoader
{
private:
	AxJVDTLabLib::AxJVLink^ m_jvlink;
	Config^ m_config;
	bool Initialize();

public:
	DataLoader(AxJVDTLabLib::AxJVLink^);
	bool Execute();
};

