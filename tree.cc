#include <iostream>  // std::cin, std::cout
#include <string>    // std::string
#include <sstream>   // std::stringstream
#include <vector>    // std::vector
#include <algorithm> // std::sort
#include <cmath>     // std::log2
#include <random>    // std::mt19937
#include <iomanip>   // std::setprecision
#include <memory>    // std::unique_ptr

namespace fdt { // flowers decision tree

namespace { // anonymous namespace; nothing outside fdt can access members
	int    max_depth;
	enum   Class { setosa, versicolor, virginica };
	enum   Feature { SL /* sepal length */, SW /* sepal width */, PL /* petal length */, PW /* petal width */ };
	double linlog(double x) { return (x == 0) ? 0 : x * std::log2(x); } // the linealogarithm function
	double I(double x, double y, double z) { return 0 - linlog(x) - linlog(y) - linlog(z); } // the information gain function
}

class Flower {
	double sl_;    // sepal length
	double sw_;    // sepal width
	double pl_;    // petal length
	double pw_;    // petal width
	Class  class_; // class

public:
	double feature(Feature f) const; // the value of Feature f for this Flower
	Class  get_class() const { return class_; }
	void   read_from(std::string const &line); // reads this Flower's data from line
};

class Node {
	std::string           feature_;
	double                threshold_;
	std::string           position_;
	std::vector<Flower>   flowers_;
	std::unique_ptr<Node> left_;
	std::unique_ptr<Node> right_;
	
	void   sort_flowers_by(Feature f);
	void   count_class(int &a, int &b, int &c) const; // number of flowers at this Node of different Classes
	double gain() const; // information gain at this Node
	void   split_node(int index); // splits this Node into left and right, splits flowers at index
	double max_gain(Feature f); // maximum possible gain at this Node for Feature f; updates all Node attributes
	int    find_best(int a, int b, int c) const; // finds the best Class representative; breaks ties randomly
	void   make_leaf();

public:
	Node(std::vector<Flower> const &new_flowers, std::string const &name) { flowers_ = new_flowers; position_ = name; }
	void   set_max_depth(int depth) const { max_depth = depth + position_.size(); }
	void   print_tree() const;
	void   build_tree();
	bool   validate_flower(Flower &f) const;
};

double Flower::feature(Feature f) const {
	switch (f) {
		case SL: return sl_;
		case SW: return sw_;
		case PL: return pl_;
		case PW: return pw_;
	}
}

void Flower::read_from(std::string const &line) {
	int c;
	char comma;
	std::stringstream ss(line);
	ss >> sl_ >> comma >> sw_ >> comma >> pl_ >> comma >> pw_ >> comma >> c;
	class_ = (Class)c;
}

void Node::sort_flowers_by(Feature f) {
	return std::sort(flowers_.begin(), flowers_.end(), 
		[f](Flower &f1, Flower &f2) -> bool {
			return f1.feature(f) < f2.feature(f);
		}
	);
}

void Node::count_class(int &a, int &b, int &c) const {
	a = b = c = 0;
	for (auto &f : flowers_) {
		switch (f.get_class()) {
			case setosa:     a++;
				break;
			case versicolor: b++;
				break;
			case virginica:  c++;
				break;
		}
	}
}

double Node::gain() const {
	int a, b, c, a1, b1, c1, a2, b2, c2;
	count_class(a, b, c);
	left_->count_class(a1, b1, c1);
	right_->count_class(a2, b2, c2);
	int total  = a+b+c;
	int total1 = a1+b1+c1;
	int total2 = a2+b2+c2;
	if (total1 == 0 || total2 == 0) return 0; // Note: total = total1 + total2
	else return I((double)a/total, (double)b/total, (double)c/total) 
		- ((double)total1/total) * I((double)a1/total1, (double)b1/total1, (double)c1/total1) 
		- ((double)total2/total) * I((double)a2/total2, (double)b2/total2, (double)c2/total2);
}

void Node::split_node(int index) {
	std::vector<Flower> lflowers(flowers_.begin(), flowers_.begin() + index);
	std::vector<Flower> rflowers(flowers_.begin() + index, flowers_.end());
	left_  = std::make_unique<Node>(lflowers, position_ + "L");
	right_ = std::make_unique<Node>(rflowers, position_ + "R");
}

double Node::max_gain(Feature f) {
	sort_flowers_by(f);
	int index = 1;
	double cur_max = 0;
	for (int split_point = 1; split_point < flowers_.size(); split_point++) {
		for (; split_point < flowers_.size(); split_point++) {
			if (flowers_[split_point-1].feature(f) != flowers_[split_point].feature(f)) {
				break;
			}
		}
		split_node(split_point);
		double candidate = gain();
		if (candidate > cur_max) {
			cur_max = candidate;
			index = split_point;
		}
		split_point++;
	}

	split_node(index);
	threshold_ = (flowers_[index-1].feature(f) + flowers_[index].feature(f)) / 2;
	switch (f) {
		case SL: feature_ = "SL";
			break;
		case SW: feature_ = "SW";
			break;
		case PL: feature_ = "PL";
			break;
		case PW: feature_ = "PW";
			break;
	}
	return cur_max;
}

int Node::find_best(int a, int b, int c) const {
	std::random_device rd;
	std::mt19937 g(rd());
	std::uniform_int_distribution<int> d2(0, 2);
	std::uniform_int_distribution<int> d1(0, 1);
	if (a == b && b == c)     return d2(g);
	else if (a == b && b > c) return d1(g);
	else if (b == c && c > a) return 1 + d1(g);
	else if (c == a && a > b) return 2 * d1(g);
	else if (a == b)          return 2;
	else if (b == c)          return 0;
	else if (c == a)          return 1;
	else                      return (b>a && b>c) + 2*(c>a && c>b);
}

void Node::make_leaf() {
	left_.reset();
	right_.reset();
	int a, b, c;
	count_class(a, b, c);
	feature_ = std::to_string(find_best(a, b, c));
	threshold_ = 0;
}

void Node::print_tree() const {
	std::cout << std::endl << "Node ID:\t" << feature_ << std::endl;
	std::cout << std::setprecision(2) << std::fixed <<  "Threshold:\t" << threshold_ << std::endl;
	std::cout << "Position:\t" << (position_ == "" ? "Root" : position_) << std::endl;
	for (auto &f : flowers_) {
		std::cout << std::setprecision(1) << std::fixed << f.feature(SL) << ',' << f.feature(SW) << ',' 
			<< f.feature(PL) << ',' << f.feature(PW) << ',' << (double)f.get_class() << std::endl;
	}
	if (left_)  left_->print_tree();
	if (right_) right_->print_tree();
}

void Node::build_tree() {
	for (int i = 0; i <= flowers_.size()-1; i++) {
		if (i == flowers_.size()-1 /* all examples same */|| max_depth == position_.size() /* reached maximum depth */) {
			make_leaf();
			return;
		}
		if (flowers_[i].get_class() != flowers_[i+1].get_class()) break;
	}

	double gainSL = max_gain(SL);
	double gainSW = max_gain(SW);
	double gainPL = max_gain(PL);
	double gainPW = max_gain(PW);
	double gain = 0;
	if (gainSL == 0 && gainSW == 0 && gainPL == 0 && gainPW == 0) { // no feature left
		make_leaf();
		return;
	} else if (gainSL >= gainSW && gainSL >= gainPL && gainSL >= gainPW) {
		gain = max_gain(SL);
	} else if (gainSW >= gainSL && gainSW >= gainPL && gainSW >= gainPW) {
		gain = max_gain(SW);
	} else if (gainPL >= gainSL && gainPL >= gainSW && gainPL >= gainPW) {
		gain = max_gain(PL);
	} else {
		gain = max_gain(PW);
	}

	left_->build_tree();
	right_->build_tree();
}

bool Node::validate_flower(Flower &f) const {
	if (feature_ == "SL") {
		if (f.feature(SL) < threshold_) return left_->validate_flower(f);
		else return right_->validate_flower(f);
	} else if (feature_ == "SW") {
		if (f.feature(SW) < threshold_) return left_->validate_flower(f);
		else return right_->validate_flower(f);
	} else if (feature_ == "PL") {
		if (f.feature(PL) < threshold_) return left_->validate_flower(f);
		else return right_->validate_flower(f);
	} else if (feature_ == "PW") {
		if (f.feature(PW) < threshold_) return left_->validate_flower(f);
		else return right_->validate_flower(f);
	} else {
		return std::to_string(f.get_class()) == feature_;
	}
}

} // namespace fdt

int main(int argc, char **argv) {
	std::vector<fdt::Flower> tflowers;
	std::string line;
	while (getline(std::cin, line)) {
		fdt::Flower f;
		f.read_from(line);
		tflowers.push_back(f);
	}

	int vset_begin = atoi(argv[1]), vset_end = atoi(argv[2]);
	std::vector<fdt::Flower> vflowers(tflowers.begin() + vset_begin, tflowers.begin() + vset_end);
	tflowers.erase(tflowers.begin() + vset_begin, tflowers.begin() + vset_end);

	fdt::Node ttree(tflowers, (argc > 4 ? argv[4] : ""));
	ttree.set_max_depth(atoi(argv[3]));
	ttree.build_tree();

	int correctt = 0;
	for (auto &f : tflowers) {
		if (ttree.validate_flower(f)) {
			correctt++;
		}
	}

	int correctv = 0;
	for (auto &f : vflowers) {
		if (ttree.validate_flower(f)) {
			correctv++;
		}
	}

	std::cout << "Validation Set:\tFlowers " << vset_begin << " to " << vset_end-1 << std::endl;
	std::cout << "Maximum Depth:\t" << argv[3] << std::endl;
	ttree.print_tree();
	std::cout << "\nTrain Accuracy:\t" << correctt << '/' << tflowers.size() << std::endl;
	std::cout << "Test Accuracy:\t" << correctv << '/' << vflowers.size() << std::endl;
}
