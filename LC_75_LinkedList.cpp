#include<bits/stdc++.h>
#include "print_arr.h"
using namespace std;
class Node{
    public:
        int val;
        Node* next;
        Node(int data){
            this->val = data;
            this->next = nullptr;
        }
};
Node *array_to_LL(vector<int> &ar)
{
    if (ar.size() == 0)
        return nullptr;
    Node *head = new Node(ar[0]);
    Node *cur = head;
    for (int i = 1; i < ar.size(); i++)
    {
        cur->next = new Node(ar[i]);
        cur = cur->next;
    }
    return head;
}
class del_mid_node{
    public:
        Node* del(Node* head){
            if(!head->next) return nullptr;
            Node *slow = head,*fast = head->next->next;
            while(fast && fast->next){
                fast = fast->next->next;
                slow = slow->next;
            }
            slow->next = slow->next->next;
            return head;
           // return slow->next;
        }
};
class Reverse {
    public:
    Node* rev(Node* head){
        Node *cur = head,*prev = nullptr;
        while(cur){
            Node* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }
};
class odd_even_LL {
public:
    Node *oddEvenList(Node *head){
         if(!head) return NULL;
        Node* odd = head,*evenHead = head->next,*even = head->next;
        while(even && even->next){
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = evenHead;
        return head;
    }
};
class maximum_twin_sum {
public:
    // 1->2->3->4   even size => twins_idx = (0,3) (1,2) return maxtwin sum
    int pairsum(Node* head){
        //find mid 
        Node* slow = head,*fast = head;
        while(fast&& fast->next){
            slow = slow->next;
            fast = fast->next->next;
        }
        Node *mid = slow;

        //reverse list
        Node *prev = NULL,*cur = mid;
        while(cur){
            Node* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        Node* st = head,*end = prev;
        int maxsum = 0;
        while(end){
            int cursum = st->val + end->val;
            maxsum = max(maxsum,cursum);
            st = st->next;
            end = end->next;
        }
        return maxsum;
    }
};
int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
vector<int> ar = {4,2,2,3};
Node* head = array_to_LL(ar);
maximum_twin_sum s;
cout<<s.pairsum(head);
/*while(ans){
    cout<<ans->val<<" ";
    ans = ans->next;
}*/

return 0;
}