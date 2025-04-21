#pragma once
ref class Config
{
public:
	System::String^ Sid;
	Config();
	static Config^ LoadFromFile(System::String^);
};

