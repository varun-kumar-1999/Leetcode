#include<bits/stdc++.h>
#include "print_arr.h"
using namespace std;
class Stack{
    public:
    vector<int> asteroidCollision(vector<int> &asteroids){
        stack<int> s;
        for(int i:asteroids){
            if(i > 0 || s.empty()) s.push(i);
            else{
                // -ve > +ve
                while(!s.empty() && s.top() < -i && s.top() > 0){
                    s.pop();
                }   
                if(!s.empty() && s.top() == abs(i)) s.pop();    // +ve == -ve
                else if(s.empty() || s.top() < 0){ // empty stack && -ve value
                    s.push(i);
                }
            }
        }
        vector<int> ans;
        while (!s.empty()){
            ans.push_back(s.top());
            s.pop();
        }
        reverse(begin(ans),end(ans));
        return ans;
    }
};

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    Stack s;
    vector<int> ar = {10,2,-5};
    vector<int>ans = s.asteroidCollision(ar);
    print_vec(ans);
    return 0;
}