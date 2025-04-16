#pragma once

#include <string>
#include <array>

namespace JVData
{

	//////////////////// 共通クラス ////////////////////

	// 年月日
	class YMD
	{
	public:
		std::string Year;  // 年
		std::string Month; // 月
		std::string Day;   // 日
		YMD() : Year(""), Month(""), Day("") {}
		YMD(std::string ymd)
		{
			Year = ymd.substr(0, 4);
			Month = ymd.substr(4, 2);
			Day = ymd.substr(6, 2);
		}
	};

	// 時分秒
	class HMS
	{
	public:
		std::string Hour;	// 時
		std::string Minute; // 分
		std::string Second; // 秒
		HMS() : Hour(""), Minute(""), Second("") {}
		HMS(std::string hms)
		{
			Hour = hms.substr(0, 2);
			Minute = hms.substr(2, 2);
			Second = hms.substr(4, 2);
		}
	};

	// 時分
	class HM
	{
	public:
		std::string Hour;	// 時
		std::string Minute; // 分
		HM() : Hour(""), Minute("") {}
		HM(std::string hm)
		{
			Hour = hm.substr(0, 2);
			Minute = hm.substr(2, 2);
		}
	};

	// 月日時分
	class MDHM
	{
	public:
		std::string Month;	// 月
		std::string Day;	// 日
		std::string Hour;	// 時
		std::string Minute; // 分
		MDHM() : Month(""), Day(""), Hour(""), Minute("") {}
		MDHM(std::string mdhm)
		{
			Month = mdhm.substr(0, 2);
			Day = mdhm.substr(2, 2);
			Hour = mdhm.substr(4, 2);
			Minute = mdhm.substr(6, 2);
		}
	};

	// レコードヘッダ
	class RECORD_ID
	{
	public:
		std::string RecordSpec; // レコード種別
		std::string DataKubun;	// データ区分
		YMD MakeDate;			// データ作成年月日
		RECORD_ID() : RecordSpec(""), DataKubun(""), MakeDate() {}
		RECORD_ID(std::string record_id)
		{
			RecordSpec = record_id.substr(0, 2);
			DataKubun = record_id.substr(2, 1);
			MakeDate = YMD(record_id.substr(3, 8));
		}
	};

	// 競走識別情報
	class RACE_ID
	{
	public:
		std::string Year;	  // 開催年
		std::string MonthDay; // 開催月日
		std::string JyoCD;	  // 競馬場コード
		std::string Kaiji;	  // 開催回[第N回]
		std::string Nichiji;  // 開催日目[N日目]
		std::string RaceNum;  // レース番号
		RACE_ID() : Year(""), MonthDay(""), JyoCD(""), Kaiji(""), Nichiji(""), RaceNum("") {}
		RACE_ID(std::string race_id)
		{
			if (race_id.size() == 16)
			{
				Year = race_id.substr(0, 4);
				MonthDay = race_id.substr(4, 4);
				JyoCD = race_id.substr(8, 2);
				Kaiji = race_id.substr(10, 2);
				Nichiji = race_id.substr(12, 2);
				RaceNum = race_id.substr(14, 2);
			}
			else if (race_id.size() == 14)
			{
				Year = race_id.substr(0, 4);
				MonthDay = race_id.substr(4, 4);
				JyoCD = race_id.substr(8, 2);
				Kaiji = race_id.substr(10, 2);
				Nichiji = race_id.substr(12, 2);
				RaceNum = "00";
			}
		}
	};

	// 着回数
	class CHAKUKAISU_INFO
	{
	public:
		std::array<std::string, 6> Chakukaisu; // 着回数
		CHAKUKAISU_INFO()
		{
			for (int i = 0; i < 6; ++i)
			{
				Chakukaisu[i] = "";
			}
		}
		CHAKUKAISU_INFO(std::string chakukaisu_info)
		{
			if (chakukaisu_info.size() == 18)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 3, 3);
				}
			}
			else if (chakukaisu_info.size() == 24)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 4, 4);
				}
			}
			else if (chakukaisu_info.size() == 30)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 5, 5);
				}
			}
			else if (chakukaisu_info.size() == 36)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 6, 6);
				}
			}
		}
	};

	// 本年・累計成績情報
	class SEI_RUIKEI_INFO
	{
	public:
		std::string SetYear;		// 設定年
		std::string HonSyokinTotal; // 本賞金合計
		std::string FukaSyokin;		// 付加賞金合計
		CHAKUKAISU_INFO ChakuKaisu; // 着回数
		SEI_RUIKEI_INFO() : SetYear(""), HonSyokinTotal(""), FukaSyokin("") {}
		SEI_RUIKEI_INFO(std::string sei_ruikei_info)
		{
			SetYear = sei_ruikei_info.substr(0, 4);
			HonSyokinTotal = sei_ruikei_info.substr(4, 10);
			FukaSyokin = sei_ruikei_info.substr(14, 10);
			ChakuKaisu = CHAKUKAISU_INFO(sei_ruikei_info.substr(24, 36));
		}
	};

	// 最近重賞勝利情報
	class SAIKIN_JYUSYO_INFO
	{
	public:
		RACE_ID SaikinJyusyoid; // 年月日場回日R
		std::string Hondai;		// 競走名本題
		std::string Ryakusyo10; // 競走名略称10字
		std::string Ryakusyo6;	// 競走名略称6字
		std::string Ryakusyo3;	// 競走名略称3字
		std::string GradeCD;	// グレードコード
		std::string SyussoTosu; // 出走頭数
		std::string KettoNum;	// 血統登録番号
		std::string Bamei;		// 馬名
		SAIKIN_JYUSYO_INFO() : Hondai(""), Ryakusyo10(""), Ryakusyo6(""), Ryakusyo3(""), GradeCD(""), SyussoTosu(""), KettoNum(""), Bamei("") {}
		SAIKIN_JYUSYO_INFO(std::string saikin_jyusyo_info)
		{
			SaikinJyusyoid = RACE_ID(saikin_jyusyo_info.substr(0, 14));
			Hondai = saikin_jyusyo_info.substr(14, 60);
			Ryakusyo10 = saikin_jyusyo_info.substr(74, 20);
			Ryakusyo6 = saikin_jyusyo_info.substr(94, 12);
			Ryakusyo3 = saikin_jyusyo_info.substr(106, 6);
			GradeCD = saikin_jyusyo_info.substr(112, 1);
			SyussoTosu = saikin_jyusyo_info.substr(113, 2);
			KettoNum = saikin_jyusyo_info.substr(115, 10);
			Bamei = saikin_jyusyo_info.substr(125, 36);
		}
	};

	// 本年・前年・累計成績情報
	class HON_ZEN_RUIKEISEI_INFO
	{
	public:
		std::string SetYear;							// 設定年
		std::string HonSyokinHeichi;					// 平地本賞金合計
		std::string HonSyokinSyogai;					// 障害本賞金合計
		std::string FukaSyokinHeichi;					// 平地付加賞金合計
		std::string FukaSyokinSyogai;					// 障害付加賞金合計
		CHAKUKAISU_INFO ChakuKaisuHeichi;				// 平地着回数
		CHAKUKAISU_INFO ChakuKaisuSyogai;				// 障害着回数
		std::array<CHAKUKAISU_INFO, 20> ChakuKaisuJyo;	// 競馬場別着回数
		std::array<CHAKUKAISU_INFO, 6> ChakuKaisuKyori; // 距離別着回数
		HON_ZEN_RUIKEISEI_INFO() : SetYear(""), HonSyokinHeichi(""), HonSyokinSyogai(""), FukaSyokinHeichi(""), FukaSyokinSyogai("") {}
		HON_ZEN_RUIKEISEI_INFO(std::string hon_zen_ruikei_info)
		{
			SetYear = hon_zen_ruikei_info.substr(0, 4);
			HonSyokinHeichi = hon_zen_ruikei_info.substr(4, 10);
			HonSyokinSyogai = hon_zen_ruikei_info.substr(14, 10);
			FukaSyokinHeichi = hon_zen_ruikei_info.substr(24, 10);
			FukaSyokinSyogai = hon_zen_ruikei_info.substr(34, 10);
			ChakuKaisuHeichi = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(44, 36));
			ChakuKaisuSyogai = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(80, 36));
			for (int i = 0; i < 20; ++i)
			{
				ChakuKaisuJyo[i] = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(116 + i * 36, 36));
			}
			for (int i = 0; i < 6; ++i)
			{
				ChakuKaisuKyori[i] = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(116 + i * 36 + 720, 36));
			}
		}
	};

	// レース情報
	class RACE_INFO
	{
	public:
		std::string YoubiCD;	// 曜日コード
		std::string TokuNum;	// 特別競走番号
		std::string Hondai;		// 競走名本題
		std::string Fukudai;	// 競走名副題
		std::string Kakko;		// 競走名カッコ内
		std::string HondaiEng;	// 競走名本題欧字
		std::string FukudaiEng; // 競走名副題欧字
		std::string KakkoEng;	// 競走名カッコ内欧字
		std::string Ryakusyo10; // 競走名略称10字
		std::string Ryakusyo6;	// 競走名略称6字
		std::string Ryakusyo3;	// 競走名略称3字
		std::string Kubun;		// 競走名区分
		std::string Nkai;		// 重賞回次[第N回]
		RACE_INFO() : YoubiCD(""), TokuNum(""), Hondai(""), Fukudai(""), Kakko(""), HondaiEng(""), FukudaiEng(""), KakkoEng(""), Ryakusyo10(""), Ryakusyo6(""), Ryakusyo3(""), Kubun(""), Nkai("") {}
		RACE_INFO(std::string race_info)
		{
			YoubiCD = race_info.substr(0, 1);
			TokuNum = race_info.substr(1, 4);
			Hondai = race_info.substr(5, 60);
			Fukudai = race_info.substr(65, 60);
			Kakko = race_info.substr(125, 60);
			HondaiEng = race_info.substr(185, 120);
			FukudaiEng = race_info.substr(305, 120);
			KakkoEng = race_info.substr(425, 120);
			Ryakusyo10 = race_info.substr(545, 20);
			Ryakusyo6 = race_info.substr(565, 12);
			Ryakusyo3 = race_info.substr(577, 6);
			Kubun = race_info.substr(583, 1);
			Nkai = race_info.substr(584, 3);
		}
	};

	// 天候・馬場状態
	class TENKO_BABA_INFO
	{
	public:
		std::string TenkoCD;	// 天候コード
		std::string SibaBabaCD; // 芝馬場状態コード
		std::string DirtBabaCD; // ダート馬場状態コード
		TENKO_BABA_INFO() : TenkoCD(""), SibaBabaCD(""), DirtBabaCD("") {}
		TENKO_BABA_INFO(std::string tenko_baba_info)
		{
			TenkoCD = tenko_baba_info.substr(0, 1);
			SibaBabaCD = tenko_baba_info.substr(1, 1);
			DirtBabaCD = tenko_baba_info.substr(2, 1);
		}
	};

	// 競走条件コード
	class RACE_JYOKEN
	{
	public:
		std::string SyubetuCD;				 // 競走種別コード
		std::string KigoCD;					 // 競走記号コード
		std::string JyuryoCD;				 // 重量種別コード
		std::array<std::string, 5> JyokenCD; // 競走条件コード
		RACE_JYOKEN() : SyubetuCD(""), KigoCD(""), JyuryoCD("")
		{
			for (int i = 0; i < 5; ++i)
			{
				JyokenCD[i] = "";
			}
		}
		RACE_JYOKEN(std::string race_jyoken)
		{
			SyubetuCD = race_jyoken.substr(0, 2);
			KigoCD = race_jyoken.substr(2, 3);
			JyuryoCD = race_jyoken.substr(5, 1);
			for (int i = 0; i < 5; ++i)
			{
				JyokenCD[i] = race_jyoken.substr(6 + i * 3, 3);
			}
		}
	};

	// 騎手変更情報
	class JC_INFO
	{
	public:
		std::string Futan;	   // 負担重量
		std::string KisyuCode; // 騎手コード
		std::string KisyuName; // 騎手名
		std::string MinaraiCD; // 騎手見習コード
		JC_INFO() : Futan(""), KisyuCode(""), KisyuName(""), MinaraiCD("") {}
		JC_INFO(std::string jc_info)
		{
			Futan = jc_info.substr(0, 3);
			KisyuCode = jc_info.substr(3, 5);
			KisyuName = jc_info.substr(8, 34);
			MinaraiCD = jc_info.substr(42, 1);
		}
	};

	// 発走時刻変更情報
	class TC_INFO
	{
	public:
		std::string Ji;	 // 時
		std::string Fun; // 分
		TC_INFO() : Ji(""), Fun("") {}
		TC_INFO(std::string tc_info)
		{
			Ji = tc_info.substr(0, 2);
			Fun = tc_info.substr(2, 2);
		}
	};

	// コース変更情報
	class CC_INFO
	{
	public:
		std::string Kyori;	 // 距離
		std::string TruckCD; // トラックコード
		CC_INFO() : Kyori(""), TruckCD("") {}
		CC_INFO(std::string cc_info)
		{
			Kyori = cc_info.substr(0, 4);
			TruckCD = cc_info.substr(4, 2);
		}
	};
}