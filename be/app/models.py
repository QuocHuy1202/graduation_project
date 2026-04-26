from sqlalchemy import Column, Integer, String, Text, DECIMAL, Boolean, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    original_user_id = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(100), unique=True, nullable=False)
    title = Column(Text)
    main_category = Column(String(255))
    price = Column(DECIMAL(10, 2))
    average_rating = Column(DECIMAL(3, 2))
    rating_number = Column(Integer)
    store = Column(String(255))
    
    # Dữ liệu phức tạp lưu dạng JSONB
    features = Column(JSONB)
    description = Column(JSONB)
    images = Column(JSONB)
    videos = Column(JSONB)
    categories = Column(JSONB) # Bác có thể lưu list tên danh mục vào đây cho nhanh
    details = Column(JSONB)
    bought_together = Column(JSONB)

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    rating = Column(DECIMAL(2, 1))
    title = Column(String(500))
    text = Column(Text)
    images = Column(JSONB)
    review_timestamp = Column(BigInteger)
    verified_purchase = Column(Boolean)
    helpful_vote = Column(Integer, default=0)