import pytest
from horguesc.database.sql_builder import SQLBuilder

class TestSQLBuilder:
    """SQLBuilderクラスのテスト"""

    def test_basic_select(self):
        """基本的なSELECT文のテスト"""
        builder = SQLBuilder("users")
        query, params = builder.build()
        
        assert query == "SELECT * FROM users"
        assert params == []

    def test_select_columns(self):
        """特定のカラムを選択するSELECT文のテスト"""
        builder = SQLBuilder("users")
        builder.select("id", "name", "email")
        query, params = builder.build()
        
        assert query == "SELECT id, name, email FROM users"
        assert params == []

    def test_select_as(self):
        """エイリアスを使用したSELECT文のテスト"""
        builder = SQLBuilder("users")
        builder.select_as("COUNT(*)", "user_count")
        query, params = builder.build()
        
        assert query == "SELECT COUNT(*) AS user_count FROM users"
        assert params == []

    def test_where_condition(self):
        """WHERE句のテスト"""
        builder = SQLBuilder("users")
        builder.where("active = TRUE")
        query, params = builder.build()
        
        assert query == "SELECT * FROM users WHERE (active = TRUE)"
        assert params == []
        
    def test_where_multiple_conditions(self):
        """複数のWHERE条件のテスト"""
        builder = SQLBuilder("users")
        builder.where("active = TRUE")
        builder.where("age > 18")
        query, params = builder.build()
        
        assert query == "SELECT * FROM users WHERE (active = TRUE) AND (age > 18)"
        assert params == []

    def test_where_between(self):
        """BETWEEN条件のテスト"""
        builder = SQLBuilder("users")
        builder.where_between("age", 18, 65)
        query, params = builder.build()
        
        assert query == "SELECT * FROM users WHERE (age BETWEEN %s AND %s)"
        assert params == [18, 65]

    def test_where_date_range(self):
        """日付範囲のテスト"""
        builder = SQLBuilder("events")
        builder.where_date_range("event_date", "2023-01-01", "2023-12-31")
        query, params = builder.build()
        
        assert query == "SELECT * FROM events WHERE (event_date >= %s) AND (event_date <= %s)"
        assert params == ["2023-01-01", "2023-12-31"]
        
    def test_where_date_range_start_only(self):
        """開始日のみの日付範囲テスト"""
        builder = SQLBuilder("events")
        builder.where_date_range("event_date", "2023-01-01", None)
        query, params = builder.build()
        
        assert query == "SELECT * FROM events WHERE (event_date >= %s)"
        assert params == ["2023-01-01"]
        
    def test_where_date_range_end_only(self):
        """終了日のみの日付範囲テスト"""
        builder = SQLBuilder("events")
        builder.where_date_range("event_date", None, "2023-12-31")
        query, params = builder.build()
        
        assert query == "SELECT * FROM events WHERE (event_date <= %s)"
        assert params == ["2023-12-31"]

    def test_where_in(self):
        """IN条件のテスト"""
        builder = SQLBuilder("products")
        builder.where_in("category_id", [1, 2, 3])
        query, params = builder.build()
        
        assert query == "SELECT * FROM products WHERE (category_id IN (%s, %s, %s))"
        assert params == [1, 2, 3]
        
    def test_where_in_empty_list(self):
        """空のリストでのIN条件のテスト"""
        builder = SQLBuilder("products")
        builder.where_in("category_id", [])
        query, params = builder.build()
        
        # 空のリストの場合はWHERE句は追加されないことを確認
        assert query == "SELECT * FROM products"
        assert params == []

    def test_join(self):
        """JOIN句のテスト"""
        builder = SQLBuilder("orders")
        builder.join("INNER JOIN users ON orders.user_id = users.id")
        query, params = builder.build()
        
        assert query == "SELECT * FROM orders INNER JOIN users ON orders.user_id = users.id"
        assert params == []

    def test_multiple_joins(self):
        """複数のJOIN句のテスト"""
        builder = SQLBuilder("orders")
        builder.join("INNER JOIN users ON orders.user_id = users.id")
        builder.join("LEFT JOIN products ON orders.product_id = products.id")
        query, params = builder.build()
        
        assert query == "SELECT * FROM orders INNER JOIN users ON orders.user_id = users.id LEFT JOIN products ON orders.product_id = products.id"
        assert params == []

    def test_group_by(self):
        """GROUP BY句のテスト"""
        builder = SQLBuilder("orders")
        builder.group_by_columns("user_id", "status")
        query, params = builder.build()
        
        assert query == "SELECT * FROM orders GROUP BY user_id, status"
        assert params == []

    def test_order_by(self):
        """ORDER BY句のテスト"""
        builder = SQLBuilder("users")
        builder.order_by_columns("last_name", "first_name")
        query, params = builder.build()
        
        assert query == "SELECT * FROM users ORDER BY last_name ASC, first_name ASC"
        assert params == []

    def test_order_by_desc(self):
        """降順のORDER BY句のテスト"""
        builder = SQLBuilder("products")
        builder.order_by_columns("price", desc=True)
        query, params = builder.build()
        
        assert query == "SELECT * FROM products ORDER BY price DESC"
        assert params == []

    def test_limit(self):
        """LIMIT句のテスト"""
        builder = SQLBuilder("users")
        builder.limit(10)
        query, params = builder.build()
        
        assert query == "SELECT * FROM users LIMIT 10"
        assert params == []

    def test_add_parameter(self):
        """パラメータ追加のテスト"""
        builder = SQLBuilder("users")
        builder.add_parameter("test_value")
        builder.where("name = %s")
        query, params = builder.build()
        
        assert query == "SELECT * FROM users WHERE (name = %s)"
        assert params == ["test_value"]

    def test_complex_query(self):
        """複雑なクエリ構築のテスト"""
        builder = SQLBuilder("products")
        builder.select("id", "name", "price")
        builder.select_as("stock_quantity", "quantity")
        builder.join("LEFT JOIN categories ON products.category_id = categories.id")
        builder.where("price > %s")
        builder.add_parameter(100)
        builder.where_between("created_at", "2023-01-01", "2023-12-31")
        builder.where_in("category_id", [1, 2, 3])
        builder.group_by_columns("category_id")
        builder.order_by_columns("price", desc=True)
        builder.limit(20)
        
        query, params = builder.build()
        
        expected_query = (
            "SELECT id, name, price, stock_quantity AS quantity "
            "FROM products "
            "LEFT JOIN categories ON products.category_id = categories.id "
            "WHERE (price > %s) AND (created_at BETWEEN %s AND %s) AND (category_id IN (%s, %s, %s)) "
            "GROUP BY category_id "
            "ORDER BY price DESC "
            "LIMIT 20"
        )
        
        assert query == expected_query
        assert params == [100, "2023-01-01", "2023-12-31", 1, 2, 3]

    def test_str_method(self):
        """__str__メソッドのテスト"""
        builder = SQLBuilder("users")
        builder.select("id", "name")
        builder.where("active = TRUE")
        
        result = str(builder)
        
        assert result == "SELECT id, name FROM users WHERE (active = TRUE)"