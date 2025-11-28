import logging
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load catalog data
CATALOG_FILE = Path("shared-data/day7_catalog.json")
ORDERS_FILE = Path("shared-data/day7_orders.json")

def load_catalog():
    """Load product catalog from JSON file"""
    try:
        with open(CATALOG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return {"store_name": "FreshMart Express", "categories": {}}

CATALOG = load_catalog()

def save_order_to_file(order_data: dict):
    """Save order to JSON file"""
    try:
        # Load existing orders or create new list
        if ORDERS_FILE.exists():
            with open(ORDERS_FILE, "r") as f:
                all_orders = json.load(f)
        else:
            all_orders = []
        
        # Add new order
        all_orders.append(order_data)
        
        # Save back to file
        with open(ORDERS_FILE, "w") as f:
            json.dump(all_orders, f, indent=2)
        
        logger.info(f"Order saved to {ORDERS_FILE} (total orders: {len(all_orders)})")
        return True
    except Exception as e:
        logger.error(f"Error saving order: {e}")
        return False

# Cart management
class CartManager:
    """Manage shopping cart for the customer"""
    
    def __init__(self):
        self.items = {}  # {product_name: {quantity, price, unit}}
        self.customer_name = None
    
    def add_item(self, product_name: str, quantity: float, price: float, unit: str):
        """Add item to cart or update quantity if already exists"""
        if product_name in self.items:
            self.items[product_name]["quantity"] += quantity
        else:
            self.items[product_name] = {
                "quantity": quantity,
                "price": price,
                "unit": unit
            }
        logger.info(f"Added to cart: {product_name} x{quantity}")
    
    def remove_item(self, product_name: str):
        """Remove item from cart"""
        if product_name in self.items:
            del self.items[product_name]
            logger.info(f"Removed from cart: {product_name}")
            return True
        return False
    
    def update_quantity(self, product_name: str, new_quantity: float):
        """Update quantity of item in cart"""
        if product_name in self.items:
            if new_quantity <= 0:
                self.remove_item(product_name)
            else:
                self.items[product_name]["quantity"] = new_quantity
                logger.info(f"Updated quantity: {product_name} -> {new_quantity}")
            return True
        return False
    
    def get_total(self):
        """Calculate total cart value"""
        total = sum(item["price"] * item["quantity"] for item in self.items.values())
        return round(total, 2)
    
    def is_empty(self):
        """Check if cart is empty"""
        return len(self.items) == 0
    
    def clear(self):
        """Clear all items from cart"""
        self.items = {}
        logger.info("Cart cleared")
    
    def get_summary(self):
        """Get formatted cart summary"""
        if self.is_empty():
            return "Cart is empty"
        
        summary = []
        for name, item in self.items.items():
            summary.append(f"{name}: {item['quantity']} {item['unit']} @ ₹{item['price']}/{item['unit']}")
        
        summary.append(f"\nTotal: ₹{self.get_total()}")
        return "\n".join(summary)


class FoodOrderingAgent(Agent):
    """Voice assistant for FreshMart Express - helps customers browse products and place orders"""
    
    def __init__(self) -> None:
        store_name = CATALOG.get("store_name", "FreshMart Express")
        
        # Build category list
        categories = list(CATALOG.get("categories", {}).keys())
        category_list = ", ".join(categories) if categories else "groceries, beverages, snacks, prepared_food"
        
        super().__init__(
            instructions=f"""You are a friendly voice assistant for {store_name}, helping customers order groceries and food.

Your role:
- Greet customers warmly and ask how you can help
- Help them browse products across categories: {category_list}
- When customers want to add items, ALWAYS use the add_to_cart tool with the exact product name
- Use add_ingredients_for when they ask for "ingredients for sandwich" or "ingredients for pasta" etc
- Modify or remove items as needed
- Provide cart summaries and calculate totals
- Complete orders when customer is ready (ask for name and delivery address)

Guidelines:
- Be conversational and helpful
- When customer says "add [item]" or "I want [item]", immediately call add_to_cart tool
- Use exact product names from the catalog (e.g., "Potato Chips", "Fresh Milk", "Whole Wheat Bread")
- Default quantity is 1 if not specified
- Always confirm each action (adding, removing, updating items)
- When browsing, show product names with prices
- For recipes/ingredients requests, use the add_ingredients_for tool
- Suggest checking the cart before placing order
- Keep responses natural and friendly

Remember: You're helping via voice, so keep it simple and clear!""",
        )
        
        # Initialize cart for this session
        self.cart = CartManager()
    
    @function_tool
    async def browse_category(self, category: str) -> str:
        """
        Browse products in a specific category.
        
        Args:
            category: Category name (groceries, beverages, snacks, prepared_food)
        """
        logger.info(f"Browsing category: {category}")
        
        categories = CATALOG.get("categories", {})
        category_lower = category.lower()
        
        if category_lower not in categories:
            available = ", ".join(categories.keys())
            return f"Category '{category}' not found. Available categories: {available}"
        
        products = categories[category_lower]
        
        if not products:
            return f"No products found in {category}"
        
        result = [f"Products in {category}:"]
        for product in products:
            brand = product.get("brand", "")
            size = product.get("size", "")
            tags = ", ".join(product.get("tags", []))
            result.append(
                f"- {product['name']} ({brand}, {size}): ₹{product['price']} - {tags}"
            )
        
        return "\n".join(result)
    
    @function_tool
    async def search_product(self, product_name: str) -> str:
        """
        Search for a specific product across all categories.
        
        Args:
            product_name: Name or partial name of the product
        """
        logger.info(f"Searching for product: {product_name}")
        
        search_lower = product_name.lower()
        found_products = []
        
        for category, products in CATALOG.get("categories", {}).items():
            for product in products:
                if search_lower in product["name"].lower():
                    brand = product.get("brand", "")
                    size = product.get("size", "")
                    found_products.append(
                        f"{product['name']} ({brand}, {size}) in {category}: ₹{product['price']}"
                    )
        
        if not found_products:
            return f"No products found matching '{product_name}'"
        
        return "Found:\n" + "\n".join(found_products)
    
    @function_tool
    async def add_ingredients_for(self, dish_or_meal: str) -> str:
        """
        Intelligently add multiple items needed for a specific dish or meal.
        For example: 'peanut butter sandwich', 'pasta', 'breakfast', 'sandwich'
        
        Args:
            dish_or_meal: The dish or meal name (e.g., 'sandwich', 'pasta', 'breakfast')
        """
        logger.info(f"Adding ingredients for: {dish_or_meal}")
        
        dish_lower = dish_or_meal.lower()
        
        # Find matching recipe from catalog
        recipes = CATALOG.get("recipes", {})
        recipe_ids = None
        
        for recipe_name, ids in recipes.items():
            if recipe_name in dish_lower or dish_lower in recipe_name:
                recipe_ids = ids
                break
        
        if not recipe_ids:
            return f"I don't have a recipe for '{dish_or_meal}'. Try browsing categories or adding specific items."
        
        # Add all items from recipe (default quantity 1)
        added_items = []
        for product_id in recipe_ids:
            # Find product in catalog by ID
            for category, products in CATALOG.get("categories", {}).items():
                for product in products:
                    if product.get("id") == product_id:
                        self.cart.add_item(
                            product["name"],
                            1,
                            product["price"],
                            product.get("size", "item")
                        )
                        added_items.append(product["name"])
                        break
        
        if not added_items:
            return f"Couldn't find ingredients for '{dish_or_meal}' in our catalog."
        
        items_str = ", ".join(added_items)
        return f"Added ingredients for {dish_or_meal}: {items_str}. Check your cart!"
    
    @function_tool
    async def add_to_cart(self, product_name: str, quantity: int) -> str:
        """
        Add product to shopping cart. Use the exact product name from browse or search results.
        
        Args:
            product_name: Exact product name (e.g., "Potato Chips", "Fresh Milk", "Whole Wheat Bread")
            quantity: Quantity to add (number of items, default 1)
        """
        logger.info(f"Adding to cart: {product_name} x{quantity}")
        
        # Find product in catalog - try exact match first
        for category, products in CATALOG.get("categories", {}).items():
            for product in products:
                if product["name"].lower() == product_name.lower():
                    size = product.get("size", "item")
                    
                    self.cart.add_item(
                        product["name"],
                        quantity,
                        product["price"],
                        size
                    )
                    
                    logger.info(f"Successfully added: {product['name']} x{quantity}")
                    return f"Added {quantity} x {product['name']} ({size}) to cart. Subtotal: ₹{product['price'] * quantity}"
        
        # Try partial match if exact match fails
        for category, products in CATALOG.get("categories", {}).items():
            for product in products:
                if product_name.lower() in product["name"].lower():
                    size = product.get("size", "item")
                    
                    self.cart.add_item(
                        product["name"],
                        quantity,
                        product["price"],
                        size
                    )
                    
                    logger.info(f"Successfully added (partial match): {product['name']} x{quantity}")
                    return f"Added {quantity} x {product['name']} ({size}) to cart. Subtotal: ₹{product['price'] * quantity}"
        
        logger.warning(f"Product not found: {product_name}")
        return f"Product '{product_name}' not found. Please browse categories or search to find the exact name."
    
    @function_tool
    async def remove_from_cart(self, product_name: str) -> str:
        """
        Remove product from shopping cart.
        
        Args:
            product_name: Name of product to remove
        """
        logger.info(f"Removing from cart: {product_name}")
        
        # Try exact match first
        if self.cart.remove_item(product_name):
            return f"Removed {product_name} from cart"
        
        # Try case-insensitive match
        for item_name in list(self.cart.items.keys()):
            if item_name.lower() == product_name.lower():
                self.cart.remove_item(item_name)
                return f"Removed {item_name} from cart"
        
        return f"{product_name} not found in cart"
    
    @function_tool
    async def update_cart_quantity(self, product_name: str, new_quantity: int) -> str:
        """
        Update quantity of product in cart.
        
        Args:
            product_name: Name of product
            new_quantity: New quantity (use 0 to remove)
        """
        logger.info(f"Updating cart: {product_name} -> {new_quantity}")
        
        # Try exact match
        if self.cart.update_quantity(product_name, new_quantity):
            if new_quantity == 0:
                return f"Removed {product_name} from cart"
            return f"Updated {product_name} quantity to {new_quantity}"
        
        # Try case-insensitive match
        for item_name in list(self.cart.items.keys()):
            if item_name.lower() == product_name.lower():
                self.cart.update_quantity(item_name, new_quantity)
                if new_quantity == 0:
                    return f"Removed {item_name} from cart"
                return f"Updated {item_name} quantity to {new_quantity}"
        
        return f"{product_name} not found in cart"
    
    @function_tool
    async def view_cart(self) -> str:
        """View all items currently in the shopping cart with total."""
        logger.info("Viewing cart")
        return self.cart.get_summary()
    
    @function_tool
    async def place_order(self, customer_name: str, delivery_address: str) -> str:
        """
        Place the order with customer details and save to JSON file.
        
        Args:
            customer_name: Customer's name
            delivery_address: Delivery address
        """
        logger.info(f"Placing order for {customer_name}")
        
        if self.cart.is_empty():
            return "Cart is empty. Please add items before placing order."
        
        # Store customer name
        self.cart.customer_name = customer_name
        
        # Generate order data
        order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"
        total = self.cart.get_total()
        timestamp = datetime.now().isoformat()
        
        # Build items list for JSON
        items_list = []
        for name, item in self.cart.items.items():
            items_list.append({
                "product_name": name,
                "quantity": item["quantity"],
                "unit": item["unit"],
                "price_per_unit": item["price"],
                "subtotal": round(item["price"] * item["quantity"], 2)
            })
        
        # Create order object
        order_data = {
            "order_id": order_id,
            "customer_name": customer_name,
            "delivery_address": delivery_address,
            "items": items_list,
            "total_amount": total,
            "timestamp": timestamp,
            "store_name": CATALOG.get("store_name", "FreshMart Express")
        }
        
        # Save to JSON file
        save_success = save_order_to_file(order_data)
        
        # Build confirmation message
        order_summary = [
            f"Order confirmed! Order ID: {order_id}",
            f"Customer: {customer_name}",
            f"Delivery Address: {delivery_address}",
            "",
            "Items ordered:"
        ]
        
        for name, item in self.cart.items.items():
            subtotal = item["price"] * item["quantity"]
            order_summary.append(
                f"- {name}: {item['quantity']} {item['unit']} = ₹{subtotal}"
            )
        
        order_summary.append(f"\nTotal Amount: ₹{total}")
        order_summary.append(f"Expected delivery: Within 2 hours")
        
        if save_success:
            order_summary.append(f"\nOrder saved successfully!")
        
        order_summary.append(f"\nThank you for shopping with {CATALOG.get('store_name', 'FreshMart Express')}!")
        
        # Clear cart after order
        self.cart.clear()
        
        return "\n".join(order_summary)



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Create main session - each agent will configure its own TTS
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Metrics
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Start with Food Ordering agent
    await session.start(
        agent=FoodOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    await ctx.connect()




if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    

